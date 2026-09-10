"""Telemetry helpers that project ARC v2 workflow results into smoke artifacts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .types import EvaluationResult, ExecutionResult, GoalHypothesis, PlanningResult, ResolvedGoal, VetDecision, WorkflowRunResult, WorkflowState, WorkflowStatus
from .evaluator import classify_v2_termination
from .compliance_checks import check_shift_a_invariants


def _has_positive_graph_evidence(graph_evidence: Any) -> bool:
    """True only when graph_evidence carries actual positive support, not
    just any non-empty container. plan_generator.py's real per-action
    evidence shape is a dict with confidence/supports/contradictions keys
    (e.g. {"confidence": 0.0, "contradictions": 64, "supports": 0} for an
    action the graph has confirmed doesn't work) -- `bool(that dict)` is
    True even though it says the opposite of "grounded." A candidate is
    only meaningfully graph-grounded when the evidence shows net-positive
    signal: some measured confidence, or more supporting than contradicting
    observations. Also handles the goal-level list-of-evidence-items shape
    (each item its own small evidence dict) by checking whether any item is
    itself positive."""
    if isinstance(graph_evidence, Mapping):
        confidence = graph_evidence.get("confidence") or 0.0
        supports = graph_evidence.get("supports") or 0
        contradictions = graph_evidence.get("contradictions") or 0
        return bool(confidence > 0 or supports > contradictions)
    if isinstance(graph_evidence, (list, tuple)):
        return any(_has_positive_graph_evidence(item) for item in graph_evidence)
    return False


def _has_graph_evidence_at_all(graph_evidence: Any) -> bool:
    """True when graph_evidence shows any history (attempts > 0), positive or negative.
    A214: complementary metric to _has_positive_graph_evidence, for near-term KPI visibility.
    This counts any candidate that the graph has seen before, regardless of net support/contradiction
    balance. Useful for a near-term "is the graph accumulating evidence" signal that CAN move
    within a single unsolved puzzle, unlike graph_grounded_decision_rate which requires net-positive."""
    if isinstance(graph_evidence, Mapping):
        attempts = graph_evidence.get("attempts") or 0
        supports = graph_evidence.get("supports") or 0
        contradictions = graph_evidence.get("contradictions") or 0
        # Any sign of history: attempts recorded, or accumulated evidence either way
        return bool(attempts > 0 or supports > 0 or contradictions > 0)
    if isinstance(graph_evidence, (list, tuple)):
        return any(_has_graph_evidence_at_all(item) for item in graph_evidence)
    return False


@dataclass(slots=True)
class ArcV2Telemetry:
    """Collect phase-level snapshots and final run artifacts for one task."""

    task_id: str
    game_id: str
    game_title: str = ""
    game_tags: tuple[str, ...] = ()
    append_snapshot: Callable[[dict[str, Any]], None] | None = None
    world_model_eval: bool = False
    started_at: float = field(default_factory=time.monotonic)
    tokens_input: int = 0
    tokens_output: int = 0
    _phase_history: list[dict[str, Any]] = field(default_factory=list)
    _cycle_index: int = 0
    _last_phase: str = "setup"
    _latest_observation: Mapping[str, Any] | None = None
    _latest_goal: ResolvedGoal | None = None
    _latest_plan: PlanningResult | None = None
    _latest_vet: VetDecision | None = None
    _latest_execution: ExecutionResult | None = None
    _latest_evaluation: EvaluationResult | None = None
    _llm_port: Any = None
    _graph_query_port: Any = None
    _phase_token_costs: dict[str, int] = field(default_factory=dict)

    def wrap_phase(self, phase_name: str, phase_callable: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            # Capture token delta from llm_port if available. The llm_port's counters
            # are updated synchronously during LLM calls within phase_callable, even
            # though telemetry.tokens_input/output are only synced to llm_port's values
            # after the entire workflow completes (in dispatch.py).
            tokens_before = (0, 0)
            if self._llm_port is not None:
                tokens_before = (self._llm_port.total_tokens_in, self._llm_port.total_tokens_out)

            result = phase_callable(*args, **kwargs)

            tokens_after = (0, 0)
            if self._llm_port is not None:
                tokens_after = (self._llm_port.total_tokens_in, self._llm_port.total_tokens_out)

            phase_token_delta = (tokens_after[0] - tokens_before[0]) + (tokens_after[1] - tokens_before[1])
            self._record_phase_result(phase_name, result, args, phase_token_delta=phase_token_delta)
            return result

        return wrapped

    def _record_phase_result(self, phase_name: str, result: Any, args: tuple[Any, ...], phase_token_delta: int = 0) -> None:
        payload = getattr(result, "payload", None)
        if phase_name == "perceive" and hasattr(payload, "observation"):
            self._latest_observation = getattr(payload, "observation", None)
        elif phase_name == "resolve" and isinstance(payload, ResolvedGoal):
            self._latest_goal = payload
        elif phase_name == "plan" and isinstance(payload, PlanningResult):
            self._latest_plan = payload
        elif phase_name == "vet" and isinstance(payload, VetDecision):
            self._latest_vet = payload
        elif phase_name == "execute" and isinstance(payload, ExecutionResult):
            self._latest_execution = payload
        elif phase_name == "evaluate" and isinstance(payload, EvaluationResult):
            self._latest_evaluation = payload

        # Accumulate phase token costs across the episode
        self._phase_token_costs[phase_name] = self._phase_token_costs.get(phase_name, 0) + phase_token_delta

        snapshot = self._phase_transition_snapshot(phase_name, result, args, phase_token_delta)
        self._phase_history.append(snapshot)
        self._emit(snapshot)

        if phase_name == "evaluate" and isinstance(payload, EvaluationResult):
            step_snapshot = self._step_snapshot(args)
            self._phase_history.append(step_snapshot)
            self._emit(step_snapshot)
            # Reset per-step phase token costs now that this step's
            # compliance count has been computed -- without this, a single
            # Shift-A violation on step N would silently re-count itself
            # into every subsequent step's compliance_violation_count for
            # the rest of the episode (the dict is keyed by phase name, and
            # `check_shift_a_invariants` doesn't know a violation was
            # already reported), inflating any future rate-based metric
            # built on this field into "steps since the first violation"
            # instead of "how many times did this actually happen."
            self._phase_token_costs = {}
            self._cycle_index += 1
            self._last_phase = phase_name

    def build_final_result(self, workflow_result: WorkflowRunResult) -> dict[str, Any]:
        final_state = self._latest_observation_state() or workflow_result.status.value
        failure_class = classify_v2_termination(workflow_result.status.value, workflow_result.reason or "")
        final_result = {
            "task_id": self.task_id,
            "game_id": self.game_id,
            "game_title": self.game_title,
            "game_tags": list(self.game_tags),
            "correct": self._is_success(workflow_result.status, final_state),
            "steps": workflow_result.state.step_index,
            "runtime_seconds": round(time.monotonic() - self.started_at, 3),
            "failure_class": failure_class,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "cost_usd": round((self.tokens_input + self.tokens_output) * 0.000001, 6),
            "final_state": final_state,
            "reason": workflow_result.reason,
            "traceback": workflow_result.traceback,
            "solve_phase_summary": self._solve_phase_summary(workflow_result.state),
            "world_model_snapshot": self._world_model_snapshot(workflow_result.state),
            "agent_execution_trace": list(self._phase_history),
            "sidequests_ledger": [],
            "arc_server_responses": [],
            "metadata": {
                "workflow_status": workflow_result.status.value,
                "completed_cycles": workflow_result.completed_cycles,
            },
        }
        if self.world_model_eval:
            final_result["world_model_live_snapshot"] = dict(final_result["world_model_snapshot"])
        return final_result

    def _phase_transition_snapshot(self, phase_name: str, result: Any, args: Sequence[Any], phase_token_delta: int = 0) -> dict[str, Any]:
        state = self._extract_state(args)
        payload = getattr(result, "payload", None)
        snapshot = {
            "snapshot_type": "phase_transition",
            "task_id": self.task_id,
            "game_id": self.game_id,
            "game_title": self.game_title,
            "game_tags": list(self.game_tags),
            "current_phase": phase_name,
            "from_phase": self._last_phase,
            "to_phase": phase_name,
            "phase": phase_name,
            "step": self._cycle_index,
            "status": getattr(result, "status", None).value if getattr(result, "status", None) else "ok",
            "reason": getattr(result, "reason", None),
            "runtime_seconds": round(time.monotonic() - self.started_at, 3),
            "workflow_step_index": getattr(state, "step_index", 0) if state is not None else 0,
            "phase_token_cost": phase_token_delta,
        }
        if phase_name == "evaluate" and isinstance(payload, EvaluationResult):
            snapshot.update(
                {
                    "decision": payload.decision.value,
                    "meaningful_progress": payload.meaningful_progress,
                    "falsification_delta": payload.falsification_delta,
                    "goal_id": self._goal_id(),
                    "action_id": self._action_id(),
                }
            )
        return snapshot

    def _step_snapshot(self, args: Sequence[Any]) -> dict[str, Any]:
        state = self._extract_state(args)
        perception = self._extract_perception(args)
        goal = self._latest_goal
        plan = self._latest_plan
        vet = self._latest_vet
        execution = self._latest_execution
        evaluation = self._latest_evaluation

        progress_tier = "flat"
        if evaluation is not None and isinstance(evaluation.metadata, Mapping):
            progress_tier = str(evaluation.metadata.get("progress_tier") or "flat")
        progress_class = "level" if progress_tier == "level" else "grid_change" if progress_tier == "grid_change" else "flat"
        progress_reward = 0.0
        if execution is not None and isinstance(execution.metadata, Mapping):
            progress_reward = float(execution.metadata.get("progress_reward", execution.metadata.get("reward", 0.0)) or 0.0)
        action_payload = {}
        if execution is not None and execution.candidate is not None and isinstance(execution.candidate.payload, Mapping):
            action_payload = dict(execution.candidate.payload)

        llm_escalated_plan = False
        graph_grounded = False
        graph_informed = False
        if execution is not None and execution.candidate is not None and isinstance(execution.candidate.metadata, Mapping):
            cand_meta = execution.candidate.metadata
            llm_escalated_plan = bool(cand_meta.get("llm_guidance"))
            graph_evidence = cand_meta.get("graph_evidence")
            graph_grounded = _has_positive_graph_evidence(graph_evidence) or bool(cand_meta.get("entity_neighborhood_grounded"))
            graph_informed = _has_graph_evidence_at_all(graph_evidence) or bool(cand_meta.get("entity_neighborhood_grounded"))

        exhaustion_source = None
        if evaluation is not None and isinstance(evaluation.metadata, Mapping):
            exhaustion_source = evaluation.metadata.get("exhaustion_source")

        capability_missing_count = 0
        hypothesis_confirm_contradict_attempted_count = 0
        goal_confidence_write_attempted_count = 0
        if self._graph_query_port is not None:
            pop = getattr(self._graph_query_port, "pop_capability_missing_count", None)
            if pop is not None:
                try:
                    capability_missing_count = pop()
                except Exception:
                    capability_missing_count = 0
            pop_hyp = getattr(self._graph_query_port, "pop_hypothesis_confirm_contradict_count", None)
            if pop_hyp is not None:
                try:
                    hypothesis_confirm_contradict_attempted_count = pop_hyp()
                except Exception:
                    hypothesis_confirm_contradict_attempted_count = 0
            pop_goal = getattr(self._graph_query_port, "pop_goal_confidence_write_count", None)
            if pop_goal is not None:
                try:
                    goal_confidence_write_attempted_count = pop_goal()
                except Exception:
                    goal_confidence_write_attempted_count = 0

        snapshot = {
            "snapshot_type": "step",
            "task_id": self.task_id,
            "game_id": self.game_id,
            "game_title": self.game_title,
            "game_tags": list(self.game_tags),
            "step": self._cycle_index + 1,
            "action_id": self._action_id(),
            "goal_id": self._goal_id(),
            "world_model_node_count": self._world_model_node_count(state),
            "world_model_edge_count": self._world_model_edge_count(state),
            "world_model_contradiction_count": int(getattr(state, "action_falsification_counts", {}).get(self._contradiction_lookup_key() or "", 0)) if state is not None else 0,
            "world_model_demotion_count": 0,
            "reasoning_skip_count": 0,
            "reasoning_escalation_count": int(bool(goal.metadata.get("llm_escalated"))) if goal is not None else 0,
            "mechanic_prior_recall_status": self._mechanic_prior_status(plan, goal),
            "mechanic_prior_count": self._mechanic_prior_count(plan),
            "mechanic_priors_used_count": self._mechanic_priors_used_count(plan),
            "active_goal_hypothesis_id": goal.selected.goal_id if goal is not None else self._goal_id(),
            "active_goal_confidence": goal.selected.confidence if goal is not None else 0.0,
            "selected_candidate_has_prediction": bool(getattr(plan, "candidate", None) and plan.candidate.expected_effect),
            "selected_candidate_prediction_effect_class": getattr(execution, "actual_effect", None) or getattr(getattr(plan, "candidate", None), "expected_effect", None),
            "selected_candidate_prediction_confidence": getattr(getattr(plan, "candidate", None), "score", 0.0) or 0.0,
            "planner_candidate_count": len(getattr(plan, "alternatives", ()) or ()) + (1 if getattr(plan, "candidate", None) else 0),
            "selected_candidate_has_falsification": bool(vet and not vet.approved),
            "reward": progress_reward,
            "progress_reward": progress_reward,
            "meaningful_progress": bool(evaluation.meaningful_progress if evaluation else False),
            "progress_class": progress_class,
            "action_effect_class": getattr(execution, "actual_effect", None) or "unknown",
            "action_x": action_payload.get("x"),
            "action_y": action_payload.get("y"),
            "decision_source": "arc_v2",
            "state": self._latest_observation_state(),
            "compliance_violation_count": self._compute_compliance_violation_count(evaluation),
            "llm_escalated_plan": llm_escalated_plan,
            "graph_grounded": graph_grounded,
            "exhaustion_source": exhaustion_source,
            "capability_missing_count": capability_missing_count,
            "hypothesis_confirm_contradict_attempted_count": hypothesis_confirm_contradict_attempted_count,
            "goal_confidence_write_attempted_count": goal_confidence_write_attempted_count,
            # A205: mirrors llm_escalated_plan/graph_grounded/exhaustion_source's
            # own "no Annatar configured -> safe default" degrade pattern --
            # WorkflowState.annatar_degraded defaults to False and is only
            # ever set True by WorkflowOrchestrator.run() after a Annatar
            # cycle actually raised/degraded (see workflow.py, annatar_signals
            # .py). getattr(..., False) also covers state being None (no
            # WorkflowState found in this phase call's args at all).
            "annatar_degraded": bool(getattr(state, "annatar_degraded", False)),
            # A251: same getattr(..., False) degrade pattern as
            # annatar_degraded above -- WorkflowState.resolve_degraded
            # defaults False and is only ever set True by
            # WorkflowOrchestrator.run() right after a resolve phase call
            # whose ResolvedGoal.degraded came back True (see
            # goal_resolver.py::_query_llm's now-caught llm_port.chat(...)
            # exception site).
            "resolve_degraded": bool(getattr(state, "resolve_degraded", False)),
            # A237: same getattr(..., False) degrade pattern as
            # annatar_degraded above -- WorkflowState.plan_degraded/
            # vet_degraded default False and are only ever set True by
            # WorkflowOrchestrator.run() right after a plan/vet phase call
            # whose PlanningResult.degraded/VetDecision.degraded came back
            # True (see plan_generator.py/plan_vetter.py). Covers both "no
            # graph configured" and "state is None" the same way.
            "plan_degraded": bool(getattr(state, "plan_degraded", False)),
            "vet_degraded": bool(getattr(state, "vet_degraded", False)),
            # A244: same getattr(..., False) degrade pattern, completing
            # A237's own explicitly-deferred evaluator.py follow-up --
            # WorkflowState.evaluate_degraded defaults False and is only
            # ever set True by WorkflowOrchestrator.run() right after an
            # evaluate phase call whose EvaluationResult.degraded came back
            # True (see evaluator.py's fetch_causal_path/
            # fetch_untested_actions except sites).
            "evaluate_degraded": bool(getattr(state, "evaluate_degraded", False)),
            # A256: same getattr(..., False) degrade pattern as the other
            # five _degraded fields above -- WorkflowState.perceive_degraded
            # defaults False and is only ever set True by
            # WorkflowOrchestrator.run() right after the perceive phase call
            # whose PerceptionSnapshot.degraded came back True (see
            # perceive.py's _ingest_snapshot except site).
            "perceive_degraded": bool(getattr(state, "perceive_degraded", False)),
            "graph_informed": graph_informed,
            # A224: the Cynefin readiness gate's own telemetry, per the
            # plan's acceptance criteria -- a real, queryable fact rather
            # than a silent budget-safety-valve failure. Same
            # getattr(..., default) degrade pattern as annatar_degraded
            # above: defaults are correct both when no readiness gate is
            # configured at all and before the gate has run its first cycle.
            "readiness_gate_partial": bool(getattr(state, "readiness_gate_partial", False)),
            "readiness_gate_entities_mapped": getattr(state, "readiness_gate_entities_mapped", None),
            "readiness_gate_entities_total": getattr(state, "readiness_gate_entities_total", None),
        }

        if evaluation is not None:
            snapshot.update(
                {
                    "decision": evaluation.decision.value,
                    "falsification_delta": evaluation.falsification_delta,
                    "failure_reason": evaluation.reason,
                }
            )
        if perception is not None:
            snapshot["grid_hash"] = perception.grid_hash
            # A219: entity-level effect-type classification (translation/
            # growth/shrink/appearance/disappearance/unchanged), computed in
            # perceive.py and carried here read-only -- not consumed by any
            # scoring/graph-write path in this card, see backlog/A219.md.
            snapshot["entity_effects"] = perception.metadata.get("entity_effects", []) if isinstance(perception.metadata, Mapping) else []
        return snapshot

    def _compute_compliance_violation_count(self, evaluation: EvaluationResult | None) -> int:
        """Count both Shift-A (deterministic phase token cost) and Shift-B violations."""
        # Shift-B violations from evaluation metadata
        shift_b_violations = evaluation.metadata.get("compliance_violations", []) if evaluation is not None and isinstance(evaluation.metadata, Mapping) else []

        # Shift-A violations from phase token costs
        shift_a_violations = check_shift_a_invariants(self._phase_token_costs)

        return len(shift_b_violations) + len(shift_a_violations)

    def _emit(self, snapshot: dict[str, Any]) -> None:
        if self.append_snapshot is not None:
            self.append_snapshot(snapshot)

    @staticmethod
    def _extract_state(args: Sequence[Any]) -> WorkflowState | None:
        for arg in args:
            if isinstance(arg, WorkflowState):
                return arg
        return None

    def _extract_perception(self, args: Sequence[Any]) -> Any:
        for arg in args:
            if hasattr(arg, "grid_hash") and hasattr(arg, "entities"):
                return arg
        return None

    def _latest_observation_state(self) -> str | None:
        if isinstance(self._latest_observation, Mapping):
            return str(self._latest_observation.get("state") or self._latest_observation.get("result_state") or "") or None
        return None

    def _goal_id(self) -> str | None:
        if self._latest_goal is not None:
            return self._latest_goal.selected.goal_id
        if self._latest_plan is not None and self._latest_plan.candidate is not None:
            return self._latest_plan.candidate.goal_id
        return None

    def _action_id(self) -> str | None:
        if self._latest_execution is not None:
            return self._latest_execution.action_id
        if self._latest_plan is not None and self._latest_plan.candidate is not None:
            return self._latest_plan.candidate.action_id
        if self._latest_vet is not None and self._latest_vet.candidate is not None:
            return self._latest_vet.candidate.action_id
        return None

    def _contradiction_lookup_key(self) -> str | None:
        # action_attempt_counts/action_falsification_counts are bookkept by
        # book_id (e.g. "ACTION6@x,y") for coordinate-targeted actions, not
        # the base action_id _action_id() returns — prefer book_id so this
        # actually matches an entry in those dicts for ACTION6.
        execution = self._latest_execution
        if execution is not None and execution.candidate is not None and isinstance(execution.candidate.metadata, Mapping):
            book_id = execution.candidate.metadata.get("book_id")
            if book_id:
                return str(book_id)
        return self._action_id()

    def _mechanic_prior_status(self, plan: PlanningResult | None, goal: ResolvedGoal | None) -> str:
        if plan is None and goal is None:
            return "not_called"
        if plan is not None and plan.metadata.get("graph_records"):
            return "prior_used"
        return "zero_priors"

    def _mechanic_prior_count(self, plan: PlanningResult | None) -> int:
        if plan is not None and isinstance(plan.metadata.get("graph_records"), Sequence):
            return len(plan.metadata.get("graph_records") or [])
        return 0

    def _mechanic_priors_used_count(self, plan: PlanningResult | None) -> int:
        """Count candidates whose action_id was sourced from mechanic prior action_set fields."""
        if plan is None or plan.candidate is None:
            return 0
        count = 0
        all_candidates = [plan.candidate] + list(plan.alternatives or ())
        for candidate in all_candidates:
            meta = candidate.metadata or {}
            if meta.get("mechanic_prior_source"):
                count += 1
        return count

    def _world_model_node_count(self, state: WorkflowState | None) -> int:
        # A183: was perceived-entity/goal/plan list sizes with no connection
        # to the real graph -- now the running count of confirmed successful
        # ingest_perception/record_transition writes this episode (see
        # WorkflowState.world_model_node_writes).
        return int(getattr(state, "world_model_node_writes", 0)) if state is not None else 0

    def _world_model_edge_count(self, state: WorkflowState | None) -> int:
        # A183: was goal_evidence/graph_records list sizes with no connection
        # to the real graph -- now the running count of confirmed successful
        # record_rule_evidence writes this episode (see
        # WorkflowState.world_model_edge_writes).
        return int(getattr(state, "world_model_edge_writes", 0)) if state is not None else 0

    @staticmethod
    def _solve_phase_summary(state: WorkflowState) -> dict[str, Any]:
        return {
            "active_goal_id": state.active_goal.selected.goal_id if state.active_goal is not None else None,
            "active_goal_confidence": state.active_goal.selected.confidence if state.active_goal is not None else 0.0,
            "replan_passes": state.replan_passes,
            "no_progress_count": state.consecutive_no_progress_count,
            "action_attempt_counts": dict(state.action_attempt_counts),
            "action_falsification_counts": dict(state.action_falsification_counts),
            # Post-A206 fix (2026-08-25): visible without digging through the
            # full trace -- how many investigation anchors in a row ended
            # without ever showing meaningful_progress, as of episode end.
            "annatar_unproductive_anchor_streak": state.annatar_unproductive_anchor_streak,
            # A227: readiness-gate fields (A224/A225) -- previously absent
            # from this end-of-episode summary even though _step_snapshot
            # already carried them correctly per-step.
            "readiness_gate_resolved": state.readiness_gate_resolved,
            "readiness_gate_partial": state.readiness_gate_partial,
            "readiness_gate_entities_mapped": state.readiness_gate_entities_mapped,
            "readiness_gate_entities_total": state.readiness_gate_entities_total,
        }

    @staticmethod
    def _world_model_snapshot(state: WorkflowState) -> dict[str, Any]:
        return {
            "node_count": len(state.action_attempt_counts) + len(state.action_falsification_counts),
            "edge_count": sum(state.action_falsification_counts.values()) if state.action_falsification_counts else 0,
            "contradiction_count": sum(state.action_falsification_counts.values()) if state.action_falsification_counts else 0,
            "demotion_count": state.consecutive_no_progress_count,
        }

    @staticmethod
    def _failure_class(status: WorkflowStatus, final_state: str | None) -> str | None:
        # Legacy method kept for backward compatibility
        if status == WorkflowStatus.CRASHED:
            return "crash"
        if status == WorkflowStatus.BUDGET_EXHAUSTED:
            return "budget_exhausted"
        if status == WorkflowStatus.STALLED:
            return "stalled"
        if status == WorkflowStatus.SKIPPED:
            return "veto"
        if final_state and final_state.upper() == "WIN":
            return None
        return None

    @staticmethod
    def _is_success(status: WorkflowStatus, final_state: str | None) -> bool:
        return status == WorkflowStatus.TERMINATED and str(final_state or "").upper() == "WIN"


__all__ = ["ArcV2Telemetry"]