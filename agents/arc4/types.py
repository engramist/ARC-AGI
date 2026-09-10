"""Shared ARC v2 dataclasses and enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, Mapping, TypeVar


class WorkflowPhase(StrEnum):
    PERCEIVE = "perceive"
    READINESS_GATE = "readiness_gate"
    RESOLVE = "resolve"
    PLAN = "plan"
    VET = "vet"
    EXECUTE = "execute"
    EVALUATE = "evaluate"


class PhaseStatus(StrEnum):
    OK = "ok"
    VETO = "veto"
    TERMINATE = "terminate"
    CRASH = "crash"


class WorkflowDecision(StrEnum):
    CONTINUE = "continue"
    PIVOT = "pivot"
    TERMINATE = "terminate"


class WorkflowStatus(StrEnum):
    RUNNING = "running"
    SKIPPED = "skipped"
    TERMINATED = "terminated"
    CRASHED = "crashed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STALLED = "stalled"


@dataclass(slots=True)
class PerceivedEntity:
    kind: str
    value: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "value": self.value,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PerceivedEntity:
        return cls(
            kind=d["kind"],
            value=d["value"],
            attributes=d.get("attributes", {}),
        )


@dataclass(slots=True)
class PerceptionSnapshot:
    observation: Mapping[str, Any]
    grid_hash: str
    grid_shape: tuple[int, int] | None = None
    loop_signal: bool = False
    repeated_grid_count: int = 0
    entities: tuple[PerceivedEntity, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    # A256: True when perceive.py's `_ingest_snapshot`'s
    # `graph_query_port.ingest_perception(...)` call raised while producing
    # this PerceptionSnapshot -- the failure already falls back to
    # metadata["graph_ingestion"] == "failed" (the correct, unchanged
    # behavior), this field just makes that degradation visible instead of
    # silently absorbed, mirroring EvaluationResult.degraded (A244) /
    # ResolvedGoal.degraded (A251) / PlanningResult.degraded (A237). Stays
    # False both when graph_query_port is None (no graph configured -- not
    # a failure) and when the ingest call this cycle succeeded.
    degraded: bool = False

    def to_dict(self) -> dict:
        return {
            "observation": dict(self.observation) if isinstance(self.observation, Mapping) else self.observation,
            "grid_hash": self.grid_hash,
            "grid_shape": self.grid_shape,
            "loop_signal": self.loop_signal,
            "repeated_grid_count": self.repeated_grid_count,
            "entities": [e.to_dict() for e in self.entities],
            "metadata": self.metadata,
            "degraded": self.degraded,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PerceptionSnapshot:
        return cls(
            observation=d["observation"],
            grid_hash=d["grid_hash"],
            grid_shape=d.get("grid_shape"),
            loop_signal=d.get("loop_signal", False),
            repeated_grid_count=d.get("repeated_grid_count", 0),
            entities=tuple(PerceivedEntity.from_dict(e) for e in d.get("entities", [])),
            metadata=d.get("metadata", {}),
            degraded=d.get("degraded", False),
        )


@dataclass(slots=True)
class GoalHypothesis:
    goal_id: str
    description: str
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GoalHypothesis:
        return cls(
            goal_id=d["goal_id"],
            description=d["description"],
            confidence=d.get("confidence", 0.0),
            evidence=tuple(d.get("evidence", [])),
            metadata=d.get("metadata", {}),
        )


@dataclass(slots=True)
class ResolvedGoal:
    selected: GoalHypothesis
    alternatives: tuple[GoalHypothesis, ...] = ()
    grounding_gate_passed: bool = True
    # A251: True when goal_resolver.py::_query_llm's `llm_port.chat(...)`
    # call raised while producing this ResolvedGoal -- the escalation
    # already fell back to the pre-LLM hypothesis ranking (the correct,
    # unchanged behavior, same as an unparseable/empty LLM response), this
    # field just makes that degradation visible instead of silently
    # absorbed, mirroring PlanningResult.degraded (A237) / EvaluationResult
    # .degraded (A244) / AnnatarOutcome.degraded (A205). Stays False both
    # when llm_port is None (no LLM configured -- not a failure) and when
    # the LLM call this cycle succeeded (or was never escalated to at all).
    degraded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "selected": self.selected.to_dict(),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "grounding_gate_passed": self.grounding_gate_passed,
            "degraded": self.degraded,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ResolvedGoal:
        return cls(
            selected=GoalHypothesis.from_dict(d["selected"]),
            alternatives=tuple(GoalHypothesis.from_dict(a) for a in d.get("alternatives", [])),
            grounding_gate_passed=d.get("grounding_gate_passed", True),
            degraded=d.get("degraded", False),
            metadata=d.get("metadata", {}),
        )


@dataclass(slots=True)
class PlanCandidate:
    action_id: str
    goal_id: str | None = None
    score: float = 0.0
    rationale: str = ""
    expected_effect: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    # Structured falsifiable prediction for evaluator checks.
    # Schema: {"kind": "grid_change"|"no_change"|"level_gain"|"state_change", "confidence": float}
    predicted_outcome: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    book_id: str = ""

    def __post_init__(self) -> None:
        if not self.book_id:
            self.book_id = str(self.metadata.get("book_id") or self.action_id)

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "goal_id": self.goal_id,
            "score": self.score,
            "rationale": self.rationale,
            "expected_effect": self.expected_effect,
            "payload": self.payload,
            "predicted_outcome": self.predicted_outcome,
            "metadata": self.metadata,
            "book_id": self.book_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PlanCandidate:
        return cls(
            action_id=d["action_id"],
            goal_id=d.get("goal_id"),
            score=d.get("score", 0.0),
            rationale=d.get("rationale", ""),
            expected_effect=d.get("expected_effect"),
            payload=d.get("payload", {}),
            predicted_outcome=d.get("predicted_outcome", {}),
            metadata=d.get("metadata", {}),
            book_id=d.get("book_id", ""),
        )


@dataclass(slots=True)
class PlanningResult:
    candidate: PlanCandidate | None
    alternatives: tuple[PlanCandidate, ...] = ()
    needs_vet: bool = True
    # A237: True when any of plan_generator.py's graph_port calls
    # (fetch_per_action_evidence, fetch_rules_for_action,
    # fetch_untested_actions) raised while producing this PlanningResult --
    # the candidate scoring already fell back to partial/no graph evidence
    # (the correct, unchanged behavior), this field just makes that
    # degradation visible instead of silently absorbed, mirroring
    # AnnatarOutcome.degraded (A205) / WorkflowState.readiness_gate_partial
    # (A224). Stays False both when graph_port is None (no graph configured
    # -- not a failure) and when every graph_port call this cycle succeeded.
    degraded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "needs_vet": self.needs_vet,
            "degraded": self.degraded,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PlanningResult:
        return cls(
            candidate=PlanCandidate.from_dict(d["candidate"]) if d.get("candidate") else None,
            alternatives=tuple(PlanCandidate.from_dict(a) for a in d.get("alternatives", [])),
            needs_vet=d.get("needs_vet", True),
            degraded=d.get("degraded", False),
            metadata=d.get("metadata", {}),
        )


@dataclass(slots=True)
class VetDecision:
    approved: bool
    candidate: PlanCandidate | None = None
    reason: str = ""
    alternative: PlanCandidate | None = None
    should_replan: bool = False
    # A237: True when plan_vetter.py's _check_graph_gate or
    # _has_live_rule_evidence hit their `except` branch while producing this
    # VetDecision -- both still fail open/no-override exactly as before (this
    # field doesn't change that), it just makes a graph-unreachable vet cycle
    # visible instead of indistinguishable from "the gate genuinely found
    # nothing to object to." Mirrors PlanningResult.degraded/AnnatarOutcome
    # .degraded (A205). One combined bit for both exception sites, not split
    # into gate-vs-override reasons -- no concrete consumer needs the
    # distinction yet (see card A237's "Assumptions/defaults").
    degraded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "reason": self.reason,
            "alternative": self.alternative.to_dict() if self.alternative else None,
            "should_replan": self.should_replan,
            "degraded": self.degraded,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VetDecision:
        return cls(
            approved=d["approved"],
            candidate=PlanCandidate.from_dict(d["candidate"]) if d.get("candidate") else None,
            reason=d.get("reason", ""),
            alternative=PlanCandidate.from_dict(d["alternative"]) if d.get("alternative") else None,
            should_replan=d.get("should_replan", False),
            degraded=d.get("degraded", False),
            metadata=d.get("metadata", {}),
        )


@dataclass(slots=True)
class ExecutionResult:
    action_id: str
    candidate: PlanCandidate | None
    observation: Mapping[str, Any]
    did_progress: bool = False
    predicted_effect: str | None = None
    actual_effect: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "observation": dict(self.observation) if isinstance(self.observation, Mapping) else self.observation,
            "did_progress": self.did_progress,
            "predicted_effect": self.predicted_effect,
            "actual_effect": self.actual_effect,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExecutionResult:
        return cls(
            action_id=d["action_id"],
            candidate=PlanCandidate.from_dict(d["candidate"]) if d.get("candidate") else None,
            observation=d["observation"],
            did_progress=d.get("did_progress", False),
            predicted_effect=d.get("predicted_effect"),
            actual_effect=d.get("actual_effect"),
            metadata=d.get("metadata", {}),
        )


@dataclass(slots=True)
class EvaluationResult:
    decision: WorkflowDecision
    meaningful_progress: bool
    falsification_delta: int = 0
    reason: str = ""
    next_goal: ResolvedGoal | None = None
    # A244: True when either of evaluator.py's two graph_port calls
    # (fetch_causal_path in evaluate(), fetch_untested_actions in
    # _action_space_exhausted) raised while producing this EvaluationResult
    # -- the unchanged fallback behavior (no causal override; exhaustion
    # falls through to "threshold_only") already applied, this field just
    # makes that degradation visible instead of silently absorbed, mirroring
    # PlanningResult.degraded/VetDecision.degraded (A237) and
    # AnnatarOutcome.degraded (A205). Stays False both when graph_query_port
    # is None (no graph configured -- not a failure) and when every
    # graph_port call this cycle succeeded.
    degraded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "meaningful_progress": self.meaningful_progress,
            "falsification_delta": self.falsification_delta,
            "reason": self.reason,
            "next_goal": self.next_goal.to_dict() if self.next_goal else None,
            "degraded": self.degraded,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvaluationResult:
        return cls(
            decision=WorkflowDecision(d["decision"]),
            meaningful_progress=d["meaningful_progress"],
            falsification_delta=d.get("falsification_delta", 0),
            reason=d.get("reason", ""),
            next_goal=ResolvedGoal.from_dict(d["next_goal"]) if d.get("next_goal") else None,
            degraded=d.get("degraded", False),
            metadata=d.get("metadata", {}),
        )


@dataclass(slots=True)
class AnnatarOutcome:
    """A202: the orchestrator-facing result of one Annatar cycle (see
    agents/arc4/annatar_state_machine.py for the pure state machine this
    wraps, and agents/arc4/annatar_signals.py for the glue that produces
    this). Lives in types.py (not ports.py, matching where PlanningResult/
    VetDecision/etc. already live) and carries `decision` as a plain str
    (annatar_state_machine.AnnatarDecision's value) rather than importing
    that enum here, so the dependency direction stays one-way: annatar
    modules may import types.py, never the reverse."""

    decision: str  # one of "advance" | "repeat_deepen" | "repeat_retry" | "terminate"
    anchor_ref: Any | None = None
    anchor_type: str | None = None  # "goal" | "entity" | None
    required_action_id: str | None = None  # set only for REPEAT_RETRY -- the exact action to re-propose
    required_book_id: str | None = None
    # A205: True when any graph-client call this cycle raised (thread
    # start/resume, thread-state write, or a CycleSignals graph query) --
    # the decision above is still a valid, safe fallback decision, this
    # just makes the degradation visible in telemetry instead of silently
    # swallowed. See spec section 8.
    degraded: bool = False
    # A230: Annatar's own answer to "did all actions/entities available get
    # explored and their entities/edges recorded in the world model" --
    # None when no readiness_report was passed to run_annatar_cycle this
    # cycle (normal post-readiness-gate cycles, or no readiness gate
    # configured at all); True/False set directly by run_annatar_cycle's
    # own glue code from the readiness report's status (READY/
    # PARTIAL_FALLTHROUGH -> True, NOT_READY -> False) whenever a report was
    # passed in. workflow.py's probe-path loop branches on this field
    # instead of reading readiness_status()'s raw return value itself --
    # this is the actual authority transfer A230 delivers.
    exploration_complete: bool | None = None
    # A241: Annatar's own signal that whole-episode-futility should resume
    # the readiness-probe loop instead of terminating -- real unmapped
    # territory (a live, graph-grounded entities_mapped < entities_total
    # re-check, not the stale post-probe-phase snapshot) still remains.
    # Mirrors exploration_complete's shape: a separate field alongside the
    # raw per-anchor `decision`, not a new AnnatarDecision value, so every
    # existing outcome.decision switch site is unaffected -- workflow.py is
    # the only place that acts on this field. False (never None): unlike
    # exploration_complete, this is only ever computed on the non-probe
    # path, where "not warranted" is always a real, decidable answer. See
    # agents/arc4/annatar_signals.py::run_annatar_cycle's docstring and
    # backlog/A241.md for the full design.
    resume_mapping: bool = False

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "anchor_ref": self.anchor_ref,
            "anchor_type": self.anchor_type,
            "required_action_id": self.required_action_id,
            "required_book_id": self.required_book_id,
            "degraded": self.degraded,
            "exploration_complete": self.exploration_complete,
            "resume_mapping": self.resume_mapping,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnnatarOutcome:
        return cls(
            decision=d["decision"],
            anchor_ref=d.get("anchor_ref"),
            anchor_type=d.get("anchor_type"),
            required_action_id=d.get("required_action_id"),
            required_book_id=d.get("required_book_id"),
            degraded=d.get("degraded", False),
            exploration_complete=d.get("exploration_complete"),
            resume_mapping=d.get("resume_mapping", False),
        )


T = TypeVar("T")


@dataclass(slots=True)
class PhaseResult(Generic[T]):
    phase: WorkflowPhase
    status: PhaseStatus = PhaseStatus.OK
    payload: T | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload_dict = None
        if self.payload is not None:
            if hasattr(self.payload, "to_dict"):
                payload_dict = self.payload.to_dict()
            else:
                payload_dict = self.payload
        return {
            "phase": self.phase.value,
            "status": self.status.value,
            "payload": payload_dict,
            "reason": self.reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PhaseResult:
        return cls(
            phase=WorkflowPhase(d["phase"]),
            status=PhaseStatus(d.get("status", "ok")),
            payload=d.get("payload"),
            reason=d.get("reason"),
            metadata=d.get("metadata", {}),
        )


@dataclass(slots=True)
class WorkflowState:
    step_index: int = 0
    phase_runs: int = 0
    terminated: bool = False
    termination_state: WorkflowStatus = WorkflowStatus.RUNNING
    previous_grid_hash: str | None = None
    # A170: ephemeral cache of the actual prior grid (not just its hash) so
    # perception can diff before/after cell changes. Deliberately excluded
    # from to_dict()/from_dict() -- runtime-only, not persisted state.
    previous_grid: list[list[Any]] | None = None
    # A175: ephemeral cache of the prior step's entities (for frame-to-frame
    # correspondence matching) and a monotonic counter for minting fresh
    # correspondence ids. Deliberately excluded from to_dict()/from_dict() --
    # runtime-only, not persisted state.
    previous_entities: tuple[PerceivedEntity, ...] | None = None
    next_entity_ref: int = 0
    loop_history: list[str] = field(default_factory=list)
    loop_history_pointer: int = -1
    active_goal: ResolvedGoal | None = None
    # A243: as of this card, goal_resolver.py::_should_escalate_to_llm no
    # longer reads this field -- it was a flat, whole-episode counter with
    # no reset on goal switches, so a freshly-selected goal's first cycle
    # could inherit a "stalled" count from completely unrelated prior
    # goals' failures (confirmed live, RE86 2026-09-02). That function now
    # reads state.goal_failure_counts[<goal_id>] instead (already
    # correctly per-goal-scoped, see that field's own comment below). This
    # counter itself is still live and still updated (cycle_policy.py::
    # record_evaluation_outcome, called from workflow.py::
    # _record_evaluation_state) -- it remains the whole-episode
    # "nothing has progressed at all, regardless of which goal" signal for
    # telemetry (telemetry.py's STALL_CHECK/demotion_count) and the legacy
    # no-Annatar check_stall() termination path (temporal_workflows.py
    # mirrors the same bookkeeping). Deliberately not repurposed as a
    # second OR-condition inside _should_escalate_to_llm either: the
    # genuinely-different "has the whole episode gone nowhere across many
    # different goals" question is already answered by
    # annatar_unproductive_anchor_streak below, which routes to a more
    # drastic and more appropriate consequence (TERMINATE) than re-asking
    # the LLM for goal disambiguation help.
    consecutive_no_progress_count: int = 0
    # A236: tracks how many consecutive cycles the SAME top-two goal_id pair
    # has been ambiguous, so _should_escalate_to_llm's `ambiguous` branch can
    # stop re-asking the LLM the identical question every cycle. Deliberately
    # separate from consecutive_no_progress_count (a whole-episode "nothing
    # has progressed" signal) -- conflating the two would treat "this specific
    # pair is still ambiguous" and "the whole episode is stalled" as the same
    # fact, which they are not.
    last_ambiguous_pair: tuple[str, str] | None = None
    ambiguous_pair_streak: int = 0
    action_attempt_counts: dict[str, int] = field(default_factory=dict)
    action_falsification_counts: dict[str, int] = field(default_factory=dict)
    # Per-goal_id no-progress counter: resets to 0 whenever THIS goal's
    # evaluation shows meaningful_progress, increments otherwise (see
    # workflow.py::_record_evaluation_state). A dict miss (a goal_id never
    # seen before) naturally starts at 0. Consumed by
    # goal_resolver.py::_apply_failure_decay (confidence decay for a
    # repeatedly-failed goal) and, as of A243, also by
    # goal_resolver.py::_should_escalate_to_llm (per-goal LLM-escalation
    # gate) -- both readers get a goal's own real history, never another
    # goal's.
    goal_failure_counts: dict[str, int] = field(default_factory=dict)
    latest_veto_reason: str | None = None
    latest_veto_alternative: PlanCandidate | None = None
    replan_passes: int = 0
    crash_traceback: str | None = None
    # A183: running counts of *confirmed* successful graph writes this
    # episode (result status == "ok", not "no_changes"/"capability_missing"/
    # "error") -- node_writes for ingest_perception (GridEntity/GridSnapshot)
    # and record_transition (Transition), edge_writes for record_rule_evidence
    # (PREDICTS/CONFIRMED_BY/FALSIFIED_BY). A client-side approximation of
    # real graph growth, not an exact node/edge count -- replaces the
    # previous world_model_node_count/edge_count telemetry, which counted
    # unrelated in-memory list sizes (perceived entities, goal/plan
    # alternatives) and had no connection to the actual graph at all.
    world_model_node_writes: int = 0
    world_model_edge_writes: int = 0
    # A202: which investigation thread (if any) the trajectory Annatar is
    # currently anchored on, tracked in-process across cycles so a fresh
    # thread is only started when the previous one actually concluded
    # (SATISFIED/EXHAUSTED -> ADVANCE). Shape when set:
    # {"anchor_ref": ..., "anchor_type": "goal"|"entity", "thread_id": ...,
    # "state": "exploring", "deepening_cycle_count": 0, "already_retried": False}
    # -- "state" is one of annatar_state_machine.InvestigationState's values,
    # kept as a plain str here for the same one-way-dependency reason
    # AnnatarOutcome.decision is a str above.
    active_investigation_anchor: dict[str, Any] | None = None
    # A202: the most recent REPEAT_DEEPEN/REPEAT_RETRY outcome, consumed by a
    # later card (A203) to bias goal_resolver/plan_generator toward the
    # Annatar's chosen anchor. None whenever the last outcome was
    # advance/terminate (nothing to bias toward).
    annatar_anchor_hint: AnnatarOutcome | None = None
    # A205: whether the most recent Annatar cycle degraded (a graph-client
    # call raised during that cycle -- see AnnatarOutcome.degraded). Set by
    # WorkflowOrchestrator.run() right after invoking the `annatar` dependency,
    # each cycle, mirroring how world_model_node_writes/edge_writes are also
    # mutated after their owning phase runs and read by telemetry.py's
    # _step_snapshot on the *next* step's snapshot (same one-cycle-lagged
    # characteristic this telemetry system already has for post-evaluate
    # state mutations -- not something A205 introduces). Stays at its default
    # False for the whole episode whenever no Annatar is configured at all.
    annatar_degraded: bool = False
    # A251: mirrors annatar_degraded's exact shape for the resolve phase --
    # set by WorkflowOrchestrator.run() right after each self._dependencies
    # .resolve call, from ResolvedGoal.degraded (goal_resolver.py::
    # _query_llm's now-caught llm_port.chat(...) exception site). "Most
    # recent invocation's outcome" for the cycle, same as plan_degraded/
    # vet_degraded/evaluate_degraded (resolve can run twice in one cycle,
    # once before a replan pass and once after -- the second call's flag is
    # what ends up here).
    resolve_degraded: bool = False
    # A237: mirrors annatar_degraded's exact shape for the plan/vet phases --
    # set by WorkflowOrchestrator.run() right after each self._dependencies
    # .plan/.vet call, from PlanningResult.degraded/VetDecision.degraded.
    # "Most recent invocation's outcome" for the cycle, same as
    # annatar_degraded (plan/vet can each run twice in one cycle, once
    # before a replan pass and once after -- the second call's flag is what
    # ends up here, which is correct: it's the graph-freshness state that
    # actually produced the plan/vet decision the cycle acted on).
    plan_degraded: bool = False
    vet_degraded: bool = False
    # A244: same shape as plan_degraded/vet_degraded above, but for the
    # evaluate phase -- set by WorkflowOrchestrator.run() right after each
    # self._dependencies.evaluate call (there are two call sites: the
    # A230/A241 probe-window path and the normal goal-directed path), from
    # EvaluationResult.degraded (evaluator.py's fetch_causal_path /
    # fetch_untested_actions except sites). "Most recent invocation's
    # outcome" for the cycle, same convention as plan_degraded/vet_degraded.
    evaluate_degraded: bool = False
    # A256: same getattr(..., False) degrade pattern as annatar_degraded/
    # resolve_degraded/plan_degraded/vet_degraded/evaluate_degraded above,
    # but for the perceive phase -- set by WorkflowOrchestrator.run() right
    # after the (single, non-duplicated) self._dependencies.perceive call,
    # from PerceptionSnapshot.degraded (perceive.py's _ingest_snapshot
    # except site). Unlike plan/vet/resolve/evaluate, perceive has no
    # replan-retry duplication, so there's exactly one call site to set
    # this from each cycle.
    perceive_degraded: bool = False
    # Post-A206 fix (2026-08-25, user-directed live-smoke follow-up): how
    # many investigation-thread anchors in a row have concluded (ADVANCE)
    # without ever once registering meaningful_progress. Tracks whole-
    # episode futility across DIFFERENT anchors -- the per-anchor state
    # machine (annatar_state_machine.py) already handles "this one anchor
    # is going nowhere" via its own EXHAUSTED/RETRY transitions, but nothing
    # aggregated "I've now tried N completely different anchors and NONE of
    # them showed any sign of life" into a real whole-episode decision.
    # Confirmed live: a 60-step smoke run cycled through 4+ different goal
    # anchors, every one totally unproductive (meaningful_progress=False on
    # all 120 evaluate snapshots), and nothing noticed -- the run just burned
    # its full step budget. annatar_signals.py::run_annatar_cycle
    # increments this on every unproductive ADVANCE and resets it to 0 the
    # moment any anchor shows real progress; once it crosses
    # run_annatar_cycle's max_unproductive_anchors threshold, the Annatar
    # emits AnnatarDecision.TERMINATE (an existing, already-wired
    # workflow.py code path that decision_for_state itself was documented,
    # in A202's own review notes, as never actually producing).
    annatar_unproductive_anchor_streak: int = 0
    # A224: whether the Cynefin readiness gate has already reached a
    # terminal decision (READY/PARTIAL_FALLTHROUGH) this episode. Once set,
    # WorkflowOrchestrator.run() skips calling readiness_gate on later
    # cycles -- the gate's own graph_port.fetch_entity_neighborhood/
    # fetch_entity_history calls aren't free, and re-deciding "ready" every
    # remaining cycle of a long episode would be pure waste.
    readiness_gate_resolved: bool = False
    # A224: the budget safety valve fired -- the readiness phase consumed
    # its budget fraction without every entity mapped, and fell through to
    # the normal path with a partial world-model rather than hard-blocking.
    # Telemetered as a real, queryable fact per the plan's own acceptance
    # criteria, not left as a silent failure.
    readiness_gate_partial: bool = False
    readiness_gate_entities_mapped: int | None = None
    readiness_gate_entities_total: int | None = None
    # A241: step_index at which the most recent readiness-gate resume
    # (whole-episode-futility intercepted into a return to probing instead
    # of TERMINATE -- see AnnatarOutcome.resume_mapping and annatar_
    # signals.py::run_annatar_cycle's override block) was granted. None
    # when no resume is currently active. Consumed by arc_runtime/bundle.py's
    # readiness_gate closure to rebase readiness_status()'s elapsed-budget-
    # fraction check against what remained AT THE MOMENT of the resume,
    # instead of the stale total-episode fraction that already crossed 0.5
    # the first time PARTIAL_FALLTHROUGH fired -- otherwise the very first
    # re-check after a resume would instantly re-fall-through with zero net
    # probing (see backlog/A241.md). Reset back to None by workflow.py once
    # the resumed probe window itself concludes (readiness_gate_resolved set
    # True again), so a LATER resume (entities_mapped is still <
    # entities_total) gets its own fresh rebasing point rather than
    # inheriting a stale one.
    readiness_gate_remap_started_step_index: int | None = None

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "phase_runs": self.phase_runs,
            "terminated": self.terminated,
            "termination_state": self.termination_state.value,
            "previous_grid_hash": self.previous_grid_hash,
            "loop_history": self.loop_history,
            "loop_history_pointer": self.loop_history_pointer,
            "active_goal": self.active_goal.to_dict() if self.active_goal else None,
            "consecutive_no_progress_count": self.consecutive_no_progress_count,
            "last_ambiguous_pair": list(self.last_ambiguous_pair) if self.last_ambiguous_pair else None,
            "ambiguous_pair_streak": self.ambiguous_pair_streak,
            "action_attempt_counts": self.action_attempt_counts,
            "action_falsification_counts": self.action_falsification_counts,
            "goal_failure_counts": self.goal_failure_counts,
            "latest_veto_reason": self.latest_veto_reason,
            "latest_veto_alternative": self.latest_veto_alternative.to_dict() if self.latest_veto_alternative else None,
            "replan_passes": self.replan_passes,
            "crash_traceback": self.crash_traceback,
            "world_model_node_writes": self.world_model_node_writes,
            "world_model_edge_writes": self.world_model_edge_writes,
            "active_investigation_anchor": self.active_investigation_anchor,
            "annatar_anchor_hint": self.annatar_anchor_hint.to_dict() if self.annatar_anchor_hint else None,
            "annatar_degraded": self.annatar_degraded,
            "resolve_degraded": self.resolve_degraded,
            "plan_degraded": self.plan_degraded,
            "vet_degraded": self.vet_degraded,
            "evaluate_degraded": self.evaluate_degraded,
            "perceive_degraded": self.perceive_degraded,
            "annatar_unproductive_anchor_streak": self.annatar_unproductive_anchor_streak,
            "readiness_gate_resolved": self.readiness_gate_resolved,
            "readiness_gate_partial": self.readiness_gate_partial,
            "readiness_gate_entities_mapped": self.readiness_gate_entities_mapped,
            "readiness_gate_entities_total": self.readiness_gate_entities_total,
            "readiness_gate_remap_started_step_index": self.readiness_gate_remap_started_step_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WorkflowState:
        return cls(
            step_index=d.get("step_index", 0),
            phase_runs=d.get("phase_runs", 0),
            terminated=d.get("terminated", False),
            termination_state=WorkflowStatus(d.get("termination_state", "running")),
            previous_grid_hash=d.get("previous_grid_hash"),
            loop_history=d.get("loop_history", []),
            loop_history_pointer=d.get("loop_history_pointer", -1),
            active_goal=ResolvedGoal.from_dict(d["active_goal"]) if d.get("active_goal") else None,
            consecutive_no_progress_count=d.get("consecutive_no_progress_count", 0),
            last_ambiguous_pair=tuple(d["last_ambiguous_pair"]) if d.get("last_ambiguous_pair") else None,
            ambiguous_pair_streak=d.get("ambiguous_pair_streak", 0),
            action_attempt_counts=d.get("action_attempt_counts", {}),
            action_falsification_counts=d.get("action_falsification_counts", {}),
            goal_failure_counts=d.get("goal_failure_counts", {}),
            latest_veto_reason=d.get("latest_veto_reason"),
            latest_veto_alternative=PlanCandidate.from_dict(d["latest_veto_alternative"]) if d.get("latest_veto_alternative") else None,
            replan_passes=d.get("replan_passes", 0),
            crash_traceback=d.get("crash_traceback"),
            world_model_node_writes=d.get("world_model_node_writes", 0),
            world_model_edge_writes=d.get("world_model_edge_writes", 0),
            active_investigation_anchor=d.get("active_investigation_anchor"),
            annatar_anchor_hint=AnnatarOutcome.from_dict(d["annatar_anchor_hint"]) if d.get("annatar_anchor_hint") else None,
            annatar_degraded=d.get("annatar_degraded", False),
            resolve_degraded=d.get("resolve_degraded", False),
            plan_degraded=d.get("plan_degraded", False),
            vet_degraded=d.get("vet_degraded", False),
            evaluate_degraded=d.get("evaluate_degraded", False),
            perceive_degraded=d.get("perceive_degraded", False),
            annatar_unproductive_anchor_streak=d.get("annatar_unproductive_anchor_streak", 0),
            readiness_gate_resolved=d.get("readiness_gate_resolved", False),
            readiness_gate_partial=d.get("readiness_gate_partial", False),
            readiness_gate_entities_mapped=d.get("readiness_gate_entities_mapped"),
            readiness_gate_entities_total=d.get("readiness_gate_entities_total"),
            readiness_gate_remap_started_step_index=d.get("readiness_gate_remap_started_step_index"),
        )


@dataclass(slots=True)
class WorkflowRunResult:
    status: WorkflowStatus
    state: WorkflowState
    phase_results: list[PhaseResult[Any]] = field(default_factory=list)
    reason: str | None = None
    traceback: str | None = None
    completed_cycles: int = 0

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "state": self.state.to_dict(),
            "phase_results": [p.to_dict() for p in self.phase_results],
            "reason": self.reason,
            "traceback": self.traceback,
            "completed_cycles": self.completed_cycles,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WorkflowRunResult:
        return cls(
            status=WorkflowStatus(d["status"]),
            state=WorkflowState.from_dict(d["state"]),
            phase_results=[PhaseResult.from_dict(p) for p in d.get("phase_results", [])],
            reason=d.get("reason"),
            traceback=d.get("traceback"),
            completed_cycles=d.get("completed_cycles", 0),
        )
