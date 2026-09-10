"""Thin ARC v2 workflow orchestrator."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Mapping

from .annatar_state_machine import ReadinessStatus
from .cycle_policy import (
    check_budget,
    check_stall,
    count_base_actions,
    record_evaluation_outcome,
    stall_threshold,
    termination_from_evaluation,
    untested_remaining_actions,
)
from .ports import WorkflowDependencies
from .types import (
    EvaluationResult,
    ExecutionResult,
    GoalHypothesis,
    PhaseResult,
    PhaseStatus,
    ResolvedGoal,
    VetDecision,
    WorkflowDecision,
    WorkflowPhase,
    WorkflowRunResult,
    WorkflowState,
    WorkflowStatus,
)


def wrap_execute_with_write_ahead(execute: Any, graph_port: Any) -> Any:
    """A204 / spec section 7: bracket an ExecutePhase callable with
    write-ahead cycle recording. `execute` is called synchronously and
    returns only after the real, live ARC API call has completed -- this is
    the one place in the whole trajectory-Annatar family (A200-A206) where
    a bug can mean double-acting on non-idempotent external game state, so
    the invariant here is non-negotiable: `write_cycle` failing (any
    exception, or a missing `cycle_id`) must NEVER prevent `execute` from
    running. The durability write is a safety net around the real action,
    not a gate in front of it.

    This wrapping is applied to the `execute` *dependency* itself, at
    bundle-build time in arc_runtime/bundle.py -- the same closure-over-
    graph_port pattern every other phase (resolve/plan/reason) already
    uses, per ports.py's AnnatarPhase docstring: "WorkflowOrchestrator
    itself does not need to hold a graph_port reference." WorkflowOrchestrator
    .run() itself is therefore untouched by this card: it keeps calling
    `self._dependencies.execute(...)` at exactly the same call site as
    before, and simply gets a write-ahead-aware callable when one is wired
    in by the bundle -- so `execute` is still "bracketed" from run()'s
    point of view without giving the orchestrator a `_graph_port` attribute
    it has never held.

    `thread_id` is read from `state.active_investigation_anchor["thread_id"]`
    at call time (the field A202's `run_annatar_cycle` establishes and
    maintains across cycles) -- `state` is passed into `execute` fresh each
    cycle, so this always reflects whatever thread was active as of the end
    of the *previous* cycle's `reason` phase (or None on the very first
    cycle / whenever no investigation thread is currently open), which is
    the correct, intended semantics per A202.
    """

    def _wrapped(state: WorkflowState, perception: Any, goal: Any, vet: Any) -> PhaseResult[Any]:
        cycle_id = None
        anchor = state.active_investigation_anchor
        thread_id = anchor.get("thread_id") if anchor is not None else None
        if graph_port is not None and thread_id is not None:
            write_cycle = getattr(graph_port, "write_cycle", None)
            if write_cycle is not None:
                try:
                    write_result = write_cycle(thread_id, state.step_index, action_sent=True)
                    cycle_id = write_result.get("cycle_id") if isinstance(write_result, dict) else None
                except Exception:
                    cycle_id = None  # graph unreachable -- degrade, never block the real action

        result = execute(state, perception, goal, vet)

        if cycle_id is not None:
            confirm_cycle = getattr(graph_port, "confirm_cycle", None)
            if confirm_cycle is not None:
                try:
                    confirm_cycle(cycle_id, decision="pending", confirmed=True)
                except Exception:
                    pass  # a failed confirm write must not crash a successful execution

        return result

    return _wrapped


@dataclass(slots=True)
class WorkflowLimits:
    max_cycles: int = 10
    max_replan_passes_per_cycle: int = 1
    max_consecutive_no_progress: int = 4


class WorkflowOrchestrator:
    """Route ARC v2 phases and apply only orchestration guards."""

    def __init__(self, dependencies: WorkflowDependencies, *, limits: WorkflowLimits | None = None) -> None:
        self._dependencies = dependencies
        self._limits = limits or WorkflowLimits()

    def run(self, state: WorkflowState, observation: Mapping[str, Any]) -> WorkflowRunResult:
        phase_results: list[PhaseResult[Any]] = []
        current_observation: Mapping[str, Any] = observation

        while True:
            budget_reason = check_budget(state.step_index, self._limits.max_cycles)
            if budget_reason is not None:
                budget_result = self._route_budget_through_annatar(state, current_observation, phase_results)
                if budget_result is not None:
                    return budget_result

            cycle_vetoes = 0
            try:
                perception = self._invoke_phase("perceive", self._dependencies.perceive, state, current_observation)
                phase_results.append(perception)
                perception_payload = self._require_payload(perception, WorkflowPhase.PERCEIVE)
                # A256: mirrors state.plan_degraded/vet_degraded/
                # resolve_degraded/evaluate_degraded's own placement -- set
                # immediately after the phase call, before any branching on
                # the result. perceive has no replan-retry duplication, so
                # this is the only call site.
                state.perceive_degraded = getattr(perception_payload, "degraded", False)
                state.previous_grid_hash = perception_payload.grid_hash
                state.loop_history.append(perception_payload.grid_hash)
                state.loop_history_pointer = len(state.loop_history) - 1

                if self._dependencies.readiness_gate is not None and not state.readiness_gate_resolved:
                    readiness_result = self._invoke_phase(
                        "readiness_gate", self._dependencies.readiness_gate, state, perception_payload
                    )
                    phase_results.append(readiness_result)
                    readiness_payload = self._require_payload(readiness_result, WorkflowPhase.READINESS_GATE)
                    readiness_status = readiness_payload.get("status")
                    state.readiness_gate_entities_mapped = readiness_payload.get("entities_mapped")
                    state.readiness_gate_entities_total = readiness_payload.get("entities_total")
                    state.readiness_gate_partial = readiness_status == ReadinessStatus.PARTIAL_FALLTHROUGH

                    probe_candidate = (
                        readiness_payload.get("probe_candidate")
                        if readiness_status == ReadinessStatus.NOT_READY
                        else None
                    )
                    if probe_candidate is not None:
                        # A224 Task 5: the dedicated, deterministic probe path
                        # -- skips resolve/plan/vet entirely (no LLM
                        # escalation, no RETRY/deepening-bias machinery,
                        # which is tuned for an already-anchored
                        # investigation and would prematurely abandon
                        # anchors during broad initial mapping). A synthetic
                        # goal/vet pair wraps the probe candidate so it
                        # reaches execute/evaluate through their existing,
                        # unmodified signatures -- confirmed by direct read
                        # that neither callable's real implementation
                        # branches on `goal` beyond a metadata `goal_id`
                        # field (evaluator.py) or not at all (bundle.py's
                        # execute wrapper).
                        probe_goal = ResolvedGoal(
                            selected=GoalHypothesis(
                                goal_id="readiness_probe",
                                description="Cynefin readiness probe",
                                confidence=0.0,
                            ),
                        )
                        state.active_goal = probe_goal
                        probe_vet = VetDecision(approved=True, candidate=probe_candidate)

                        execution = self._invoke_phase(
                            "execute", self._dependencies.execute, state, perception_payload, probe_goal, probe_vet
                        )
                        phase_results.append(execution)
                        execution_payload = self._require_payload(execution, WorkflowPhase.EXECUTE)

                        evaluation = self._invoke_phase(
                            "evaluate",
                            self._dependencies.evaluate,
                            state,
                            perception_payload,
                            probe_goal,
                            execution_payload,
                        )
                        phase_results.append(evaluation)
                        evaluation_payload = self._require_payload(evaluation, WorkflowPhase.EVALUATE)
                        # A244: mirrors state.plan_degraded/vet_degraded's
                        # exact placement -- set immediately after the phase
                        # call, before any branching on the result.
                        state.evaluate_degraded = getattr(evaluation_payload, "degraded", False)

                        self._record_execution_attempt(state, execution_payload)
                        # A242: probe-phase cycles are exploratory, not
                        # goal-directed attempts, and essentially never
                        # register meaningful_progress -- letting them
                        # increment state.consecutive_no_progress_count
                        # (as they did pre-A242) inflated the count to
                        # 15-21 before goal-directed play's very first
                        # cycle even ran, forcing goal_resolver.py::_should_
                        # escalate_to_llm's under_confident branch true
                        # from cycle one regardless of genuine ambiguity.
                        # count_toward_no_progress=False leaves falsification-
                        # count bookkeeping (also done inside this call)
                        # completely unaffected -- only the no-progress
                        # count itself is scoped out, mirroring A230's own
                        # non-probe-cycles-only precedent for
                        # annatar_unproductive_anchor_streak. This block is
                        # also re-entered verbatim by an A241-granted
                        # resumed probe window (state.readiness_gate_
                        # resolved reset to False re-enters this same `if
                        # probe_candidate is not None:` block), so a
                        # resumed probe window's cycles are excluded from
                        # the count too, automatically -- no separate
                        # handling needed.
                        self._record_evaluation_state(
                            state, execution_payload, evaluation_payload, count_toward_no_progress=False
                        )
                        state.step_index += 1

                        termination = termination_from_evaluation(evaluation_payload.decision, evaluation_payload.reason)
                        if evaluation.status == PhaseStatus.TERMINATE or termination is not None:
                            return self._finish(
                                state,
                                WorkflowStatus.TERMINATED,
                                evaluation_payload.reason or "terminated",
                                phase_results,
                            )

                        # A230: route every probe cycle through the SAME
                        # self._dependencies.annatar(...) call site the
                        # normal path already uses (below), instead of
                        # `continue`ing without Annatar ever seeing this
                        # cycle happen (the gap this card fixes). The
                        # readiness-gate's own report (status/entities_
                        # mapped/entities_total, already computed above,
                        # unchanged) is passed through so Annatar's signal
                        # layer can read it -- readiness_status()'s own
                        # classification logic is untouched; only who acts
                        # on it changes.
                        #
                        # A250: `annatar` is unconditionally wired in
                        # production since A202, so this call is no longer
                        # gated behind an `is not None` check -- the
                        # no-Annatar fallback branch (byte-for-byte pre-A230
                        # behavior: always continue probing, no exhaustion
                        # check at all) was permanently dead code. See
                        # backlog/A250.md.
                        #
                        # A249: thread action_space_exhausted into Annatar
                        # via the same stall_reason channel the normal-cycle
                        # call site below uses (see backlog/A249.md) -- the
                        # probe path has no check_stall-derived stall_reason
                        # of its own, so this is purely the exhaustion flag.
                        probe_stall_reason = (
                            "action_space_exhausted"
                            if evaluation_payload.metadata.get("action_space_exhausted")
                            else None
                        )
                        outcome = self._dependencies.annatar(
                            state,
                            perception_payload,
                            execution_payload,
                            evaluation_payload,
                            stall_reason=probe_stall_reason,
                            readiness_report=readiness_payload,
                        )
                        state.annatar_degraded = outcome.degraded
                        # A230 live-verification hook: matches the
                        # existing STALL_CHECK precedent below -- a
                        # greppable, concrete record that Annatar was
                        # actually invoked for this probe cycle and what
                        # it decided, since (unlike every other phase)
                        # the annatar dependency is not wrapped by
                        # telemetry.wrap_phase and so never produces a
                        # phase_transition snapshot of its own.
                        import logging as _probe_logging
                        _probe_logging.getLogger(__name__).info(
                            "PROBE_ANNATAR decision=%s exploration_complete=%s entities_mapped=%s entities_total=%s",
                            outcome.decision,
                            outcome.exploration_complete,
                            state.readiness_gate_entities_mapped,
                            state.readiness_gate_entities_total,
                        )
                        if outcome.decision == "terminate":
                            return self._finish(state, WorkflowStatus.TERMINATED, "annatar_exhausted", phase_results)

                        current_observation = execution_payload.observation
                        if outcome.exploration_complete is True:
                            # Annatar's own outcome -- not readiness_
                            # status()'s raw return value -- says the
                            # world model is sufficiently explored. Fall
                            # through (no `continue`) into the same
                            # cycle's normal resolve/plan/vet path
                            # immediately below, instead of burning a
                            # whole extra cycle just to notice the gate
                            # resolved. The "stop gating" code just below
                            # still runs and sets
                            # state.readiness_gate_resolved = True --
                            # harmlessly idempotent when already True.
                            pass
                        else:
                            # False (still not ready) or None (Annatar
                            # produced no readiness opinion this cycle --
                            # treat conservatively, same as False): keep
                            # probing next cycle.
                            continue

                    # NOT_READY with nothing left to probe this cycle,
                    # READY/PARTIAL_FALLTHROUGH, or Annatar just reported
                    # exploration_complete=True above: stop gating and
                    # proceed through the normal path from here on.
                    # Re-invoking the gate (and its graph_port calls) every
                    # remaining cycle would just repeat the same check for
                    # no benefit.
                    state.readiness_gate_resolved = True
                    # A241: this probe window (whether the episode's
                    # original mapping pass, or a resumed one granted by
                    # AnnatarOutcome.resume_mapping below) has just
                    # concluded -- clear the resume-start marker so a LATER
                    # resume (entities_mapped is still < entities_total)
                    # gets its own fresh rebasing point in arc_runtime/
                    # bundle.py's readiness_gate closure, rather than
                    # inheriting this one's now-stale start step_index.
                    # Harmlessly a no-op (already None) on the episode's
                    # very first, never-resumed probe window.
                    state.readiness_gate_remap_started_step_index = None

                resolved_goal = self._invoke_phase("resolve", self._dependencies.resolve, state, perception_payload)
                phase_results.append(resolved_goal)
                resolved_goal_payload = self._require_payload(resolved_goal, WorkflowPhase.RESOLVE)
                state.active_goal = resolved_goal_payload
                # A251: mirrors state.annatar_degraded's exact placement --
                # set immediately after the phase call, before any branching
                # on the result.
                state.resolve_degraded = getattr(resolved_goal_payload, "degraded", False)

                planning = self._invoke_phase(
                    "plan",
                    self._dependencies.plan,
                    state,
                    perception_payload,
                    resolved_goal_payload,
                )
                phase_results.append(planning)
                planning_payload = self._require_payload(planning, WorkflowPhase.PLAN)
                # A237: mirrors state.annatar_degraded's exact placement --
                # set immediately after the phase call, before any branching
                # on the result.
                state.plan_degraded = getattr(planning_payload, "degraded", False)

                vet = self._invoke_phase(
                    "vet",
                    self._dependencies.vet,
                    state,
                    perception_payload,
                    resolved_goal_payload,
                    planning_payload,
                )
                phase_results.append(vet)
                vet_payload = self._require_payload(vet, WorkflowPhase.VET)
                state.vet_degraded = getattr(vet_payload, "degraded", False)
                if not vet_payload.approved or vet.status == PhaseStatus.VETO:
                    cycle_vetoes += 1
                    state.latest_veto_reason = vet_payload.reason or vet.reason
                    state.latest_veto_alternative = vet_payload.alternative or vet_payload.candidate
                    state.replan_passes += 1
                    if cycle_vetoes > self._limits.max_replan_passes_per_cycle:
                        veto_result = self._route_second_veto_through_annatar(state, perception_payload, current_observation, phase_results)
                        if veto_result is not None:
                            return veto_result
                        continue

                    resolved_goal = self._invoke_phase("resolve", self._dependencies.resolve, state, perception_payload)
                    phase_results.append(resolved_goal)
                    resolved_goal_payload = self._require_payload(resolved_goal, WorkflowPhase.RESOLVE)
                    state.active_goal = resolved_goal_payload
                    # A251: same placement as the initial resolve call site
                    # above -- "most recent invocation" semantics, matching
                    # plan_degraded/vet_degraded's own two-call-site
                    # precedent.
                    state.resolve_degraded = getattr(resolved_goal_payload, "degraded", False)

                    planning = self._invoke_phase(
                        "plan",
                        self._dependencies.plan,
                        state,
                        perception_payload,
                        resolved_goal_payload,
                    )
                    phase_results.append(planning)
                    planning_payload = self._require_payload(planning, WorkflowPhase.PLAN)
                    # A237: same placement as the initial plan/vet call site
                    # above -- "most recent invocation" semantics, matching
                    # annatar_degraded's own two-call-site precedent.
                    state.plan_degraded = getattr(planning_payload, "degraded", False)

                    vet = self._invoke_phase(
                        "vet",
                        self._dependencies.vet,
                        state,
                        perception_payload,
                        resolved_goal_payload,
                        planning_payload,
                    )
                    phase_results.append(vet)
                    vet_payload = self._require_payload(vet, WorkflowPhase.VET)
                    state.vet_degraded = getattr(vet_payload, "degraded", False)
                    if not vet_payload.approved or vet.status == PhaseStatus.VETO:
                        state.latest_veto_reason = vet_payload.reason or vet.reason
                        state.latest_veto_alternative = vet_payload.alternative or vet_payload.candidate
                        veto_result = self._route_second_veto_through_annatar(state, perception_payload, current_observation, phase_results)
                        if veto_result is not None:
                            return veto_result
                        continue

                execution = self._invoke_phase(
                    "execute",
                    self._dependencies.execute,
                    state,
                    perception_payload,
                    resolved_goal_payload,
                    vet_payload,
                )
                phase_results.append(execution)
                execution_payload = self._require_payload(execution, WorkflowPhase.EXECUTE)

                evaluation = self._invoke_phase(
                    "evaluate",
                    self._dependencies.evaluate,
                    state,
                    perception_payload,
                    resolved_goal_payload,
                    execution_payload,
                )
                phase_results.append(evaluation)
                evaluation_payload = self._require_payload(evaluation, WorkflowPhase.EVALUATE)
                # A244: same placement as the probe-path evaluate call site
                # above -- "most recent invocation" semantics, matching
                # plan_degraded/vet_degraded's own two-call-site precedent.
                state.evaluate_degraded = getattr(evaluation_payload, "degraded", False)

                self._record_execution_attempt(state, execution_payload)
                self._record_evaluation_state(state, execution_payload, evaluation_payload)
                state.step_index += 1

                # A202 / spec section 5: the environment's own authoritative
                # terminal signal (a real win/loss from the ARC API itself,
                # via termination_from_evaluation) stays independent and
                # short-circuits *before* Annatar runs -- "an
                # environment-terminal result doesn't need a strategic
                # opinion." This check is deliberately positioned ahead of
                # the stall/Annatar block below (moved up from its prior
                # position after the stall check) so a real terminal result
                # is never routed through Annatar logic, and Annatar
                # is never invoked once the episode is already over.
                termination = termination_from_evaluation(evaluation_payload.decision, evaluation_payload.reason)
                if evaluation.status == PhaseStatus.TERMINATE or termination is not None:
                    return self._finish(state, WorkflowStatus.TERMINATED, evaluation_payload.reason or "terminated", phase_results)

                available_actions = current_observation.get("available_actions", [])
                num_available = len(available_actions)
                # Count distinct base actions so ACTION6@x,y click targets don't
                # inflate the attempted count past the available action space.
                # `attempted` here stays a whole-episode-cumulative diagnostic
                # number (informational only); `untested` below is what
                # check_stall actually gates on and A248 fixed it to be a set
                # difference against *this cycle's* available_actions, not a
                # subtraction against this cumulative count -- action_attempt_
                # counts is never reset, so it can hold stale entries from an
                # earlier, differently-composed action-space phase (see
                # backlog/A248.md).
                num_attempted = count_base_actions(state.action_attempt_counts)
                untested_remaining = untested_remaining_actions(available_actions, state.action_attempt_counts)
                import logging as _logging
                _logging.getLogger(__name__).info(
                    "STALL_CHECK no_progress=%d, available=%d, attempted=%d, untested=%d, threshold=%d",
                    state.consecutive_no_progress_count,
                    num_available or 1,
                    num_attempted,
                    untested_remaining,
                    stall_threshold(self._limits.max_consecutive_no_progress, num_available),
                )
                stall_reason = check_stall(
                    state.consecutive_no_progress_count,
                    self._limits.max_consecutive_no_progress,
                    num_available,
                    untested_remaining,
                )

                # A202 / spec section 5: the Annatar owns the
                # advance/repeat/terminate decision. check_stall's signal
                # is folded in as one of its inputs (see
                # annatar_signals.compute_cycle_signals) instead of
                # independently ending the run.
                #
                # A250: `annatar` is unconditionally wired in production
                # since A202, so this call is no longer gated behind an
                # `is not None` check -- the no-Annatar fallback branch
                # (bare `check_stall` ending the run directly as STALLED,
                # with no arbiter) was permanently dead code. See
                # backlog/A250.md.
                #
                # A212 (visibility-only, audit conclusion): if this
                # cycle's first plan_vetter rejection was resolved by the
                # local same-cycle retry (cycle_vetoes > 0 but we still
                # reached here -- a second veto would have routed through
                # _route_second_veto_through_annatar and `continue`d
                # instead), pass its reason/alternative along too.
                # Gating on the local `cycle_vetoes` counter (not
                # directly reading state.latest_veto_reason) matters:
                # that state field is never reset, so reading it
                # unconditionally would leak a stale veto from an earlier
                # cycle into a cycle that had no veto at all. Purely
                # informational -- see CycleSignals.veto_reason's
                # docstring for why this cannot change what happens next.
                veto_reason = state.latest_veto_reason if cycle_vetoes else None
                veto_alternative_action_id = (
                    state.latest_veto_alternative.action_id
                    if cycle_vetoes and state.latest_veto_alternative is not None
                    else None
                )
                # A234: resolved_goal_payload is already in scope (set a
                # few lines above, right after the last `resolve` call
                # this cycle -- either the first resolve at the top of
                # the cycle, or the local same-cycle retry's resolve if
                # a first veto triggered one). Fold goal_resolver.py::
                # resolve()'s own already-computed per-cycle output into
                # this same existing Annatar call, mirroring A230's
                # readiness_report -- no new call site, no new routing
                # logic. `top_two_confidence_gap` re-derives the same
                # ambiguity measure goal_resolver.py::_should_escalate_
                # to_llm computes internally (selected vs. top
                # alternative's confidence), read-only here from data
                # already sitting on ResolvedGoal -- None when there is
                # no alternative to compare against.
                resolve_report = {
                    "grounding_gate_passed": resolved_goal_payload.grounding_gate_passed,
                    "llm_escalated": bool(resolved_goal_payload.metadata.get("llm_escalated")),
                    "llm_reason": resolved_goal_payload.metadata.get("llm_reason"),
                    "hypothesis_count": len(resolved_goal_payload.metadata.get("hypotheses", [])),
                    "top_two_confidence_gap": (
                        resolved_goal_payload.selected.confidence - resolved_goal_payload.alternatives[0].confidence
                        if resolved_goal_payload.alternatives
                        else None
                    ),
                }
                # A249: fold action_space_exhausted into the same
                # stall_reason channel Annatar already reads (see
                # backlog/A249.md) -- annatar_signals.py:251 only ever
                # checks `stall_reason is not None`, never its string
                # value, so this OR is safe: whichever signal fired (or
                # both) still produces the identical all_falsified=True/
                # untested_remaining=False override.
                effective_stall_reason = stall_reason or (
                    "action_space_exhausted" if evaluation_payload.metadata.get("action_space_exhausted") else None
                )
                outcome = self._dependencies.annatar(
                    state,
                    perception_payload,
                    execution_payload,
                    evaluation_payload,
                    stall_reason=effective_stall_reason,
                    veto_reason=veto_reason,
                    veto_alternative_action_id=veto_alternative_action_id,
                    resolve_report=resolve_report,
                )
                # A234 live-verification hook: mirrors A230's own
                # PROBE_ANNATAR precedent -- a greppable, concrete
                # record that resolve()'s real output actually reached
                # this call, since (unlike every other phase) the
                # annatar dependency is not wrapped by
                # telemetry.wrap_phase and so never produces a
                # phase_transition snapshot of its own.
                import logging as _resolve_logging
                _resolve_logging.getLogger(__name__).info(
                    "RESOLVE_ANNATAR grounding_gate_passed=%s llm_escalated=%s top_two_confidence_gap=%s",
                    resolve_report["grounding_gate_passed"],
                    resolve_report["llm_escalated"],
                    resolve_report["top_two_confidence_gap"],
                )
                # A205 / spec section 8: make a degraded (graph-unreachable)
                # Annatar cycle visible in telemetry rather than silently
                # swallowed. Set every cycle the Annatar actually runs, so
                # this always reflects the most recent cycle's outcome.
                state.annatar_degraded = outcome.degraded
                if outcome.resume_mapping:
                    # A241: Annatar intercepted its own whole-episode-
                    # futility override -- real unmapped territory
                    # remains (a live, graph-grounded re-check, per
                    # annatar_signals.py::run_annatar_cycle's docstring),
                    # so resume the readiness-probe loop instead of
                    # honoring what would otherwise be TERMINATE.
                    # Resetting readiness_gate_resolved alone would do
                    # nothing this cycle -- the gate's own `if` block
                    # (top of this method) already ran and moved past
                    # for THIS cycle's perception -- so `continue` routes
                    # control back to the top of the outer `while True:`
                    # loop, letting the NEXT cycle's iteration naturally
                    # re-enter that `if` block and the existing probe-
                    # path code (the `if probe_candidate is not None:`
                    # block above) instead of duplicating it here.
                    # state.active_investigation_anchor is already None
                    # (run_annatar_cycle's decision.value == "advance"
                    # branch clears it before this override ever runs)
                    # and state.annatar_unproductive_anchor_streak was
                    # already reset to 0 by run_annatar_cycle itself, so
                    # neither needs repeating here.
                    state.readiness_gate_resolved = False
                    current_observation = execution_payload.observation
                    continue
                if outcome.decision == "terminate":
                    return self._finish(state, WorkflowStatus.TERMINATED, "annatar_exhausted", phase_results)
                if outcome.decision in ("repeat_deepen", "repeat_retry"):
                    # Consumed by a later card (A203, anchor-biasing in
                    # goal_resolver/plan_generator) -- this card only
                    # needs to produce and store the hint correctly.
                    state.annatar_anchor_hint = outcome
                else:
                    state.annatar_anchor_hint = None

                current_observation = execution_payload.observation
            except Exception:
                # A211: best-effort close-out of investigation thread on crash,
                # without ever risking the original crash's traceback being masked.
                traceback_text = traceback.format_exc()
                anchor = state.active_investigation_anchor
                thread_id = anchor.get("thread_id") if isinstance(anchor, dict) else None
                # A250: `annatar` is unconditionally wired in production
                # since A202 -- this used to be a three-way AND-gate
                # (annatar configured AND thread_id AND on_crash_cleanup);
                # with `annatar` guaranteed non-None it narrows to the two
                # conditions that actually vary. See backlog/A250.md.
                if thread_id is not None and self._dependencies.on_crash_cleanup is not None:
                    try:
                        # Close out the thread's graph state before returning CRASHED.
                        # State value "exhausted" is used as an interim fit for the crash
                        # close-out case, per A211's analysis that the existing state enum
                        # has no perfect semantic match for "abnormally terminated."
                        self._dependencies.on_crash_cleanup(thread_id, "exhausted")
                    except Exception:
                        # A211 non-negotiable: cleanup failure must never mask or replace
                        # the original crash's traceback. The cleanup exception is silently
                        # swallowed here; only the original traceback is reported.
                        pass
                return self._finish(
                    state,
                    WorkflowStatus.CRASHED,
                    "crash",
                    phase_results,
                    traceback_text=traceback_text,
                )

    def _route_budget_through_annatar(
        self,
        state: WorkflowState,
        current_observation: Mapping[str, Any],
        phase_results: list[PhaseResult[Any]],
    ) -> WorkflowRunResult:
        """A209 fix (2026-08-25): `check_budget` previously ended the episode
        directly, without ever giving anything a chance to close out an open
        investigation thread's graph bookkeeping first -- another place a
        termination could happen with no cleanup. The budget ceiling itself is
        non-negotiable: this method always ends the episode as
        BUDGET_EXHAUSTED regardless of anything below.

        A252 (2026-09-04): previously ran a full synthetic Annatar cycle
        (fabricated PerceptionSnapshot/ExecutionResult/EvaluationResult
        payloads through the real `self._dependencies.annatar(...)`) purely
        to trigger its own end-of-cycle `write_thread_state` bookkeeping --
        discarding the decision it produced but keeping the graph write,
        which was computed from data that doesn't reflect anything real and
        could land on a specific, misleadingly-precise `InvestigationState`
        value. It also risked a real LLM call (`resolve_llm_vote`) if the
        open thread happened to be AWAITING_LLM, spending real cost on a
        decision nobody reads. Replaced with the same direct close-out A211
        already uses for the crash path: call `on_crash_cleanup(thread_id,
        ...)` directly, best-effort, no synthetic Annatar cycle, no risk of
        a misleading graph write or a wasted LLM call. See backlog/A252.md.

        Every branch of this method returns a real WorkflowRunResult -- there
        is no path that returns None. The close-out call's own outcome
        (success, no-op, or exception) is never inspected either: the budget
        ceiling is non-negotiable regardless of it, so the method doesn't
        depend on (and doesn't assert) any particular close-out result. This
        is what makes the hard-ceiling guarantee structural -- see
        test_annatar_response_does_not_override_budget for the proof (a
        Annatar mock returning "advance" still ends the episode as
        BUDGET_EXHAUSTED with zero further phases invoked)."""
        # First iteration has no prior cycle to report; end immediately.
        if state.step_index == 0:
            return self._finish(state, WorkflowStatus.BUDGET_EXHAUSTED, "budget_exhausted", phase_results)

        # For step_index > 0, best-effort close out any open investigation
        # thread directly -- mirrors the crash handler's own pattern exactly
        # (see workflow.py's `except Exception:` block above). Cleanup
        # failure must never prevent the real BUDGET_EXHAUSTED result from
        # returning.
        anchor = state.active_investigation_anchor
        thread_id = anchor.get("thread_id") if isinstance(anchor, dict) else None
        if thread_id is not None and self._dependencies.on_crash_cleanup is not None:
            try:
                self._dependencies.on_crash_cleanup(thread_id, "exhausted")
            except Exception:
                pass  # best-effort, same non-negotiable as the crash path

        return self._finish(state, WorkflowStatus.BUDGET_EXHAUSTED, "budget_exhausted", phase_results)

    def _route_second_veto_through_annatar(
        self,
        state: WorkflowState,
        perception_payload: Any,
        current_observation: Mapping[str, Any],
        phase_results: list[PhaseResult[Any]],
    ) -> WorkflowRunResult | None:
        """Post-A206 fix (2026-08-25): a double veto previously ended the
        episode directly, without ever giving the Annatar a say -- the one
        place the "one agent that sees everything end-to-end" was
        structurally excluded from a real strategic decision ("the safety
        layer just rejected our plan twice, what now"). `vet` fires before
        `execute`/`evaluate`, so there's no real ExecutionResult/
        EvaluationResult for this cycle -- a synthetic "nothing was
        attempted" pair is fed to the Annatar instead
        (execution.candidate=None, meaningful_progress=False), plus
        stall_reason="second_veto" (reusing the existing stall-fold
        mechanism from compute_cycle_signals rather than inventing a
        parallel one). A repeated-double-veto pathological case is bounded
        by the same whole-episode-futility streak
        (annatar_signals.run_annatar_cycle) built alongside this fix --
        no new termination logic needed for that case specifically.

        Returns a WorkflowRunResult if the episode should end now (the
        Annatar said "terminate"); None if the caller should `continue` the
        outer loop instead, letting the Annatar's decision (a fresh anchor,
        or the same one) drive the next cycle."""
        synthetic_execution = ExecutionResult(action_id="", candidate=None, observation=current_observation, metadata={})
        synthetic_evaluation = EvaluationResult(
            decision=WorkflowDecision.CONTINUE,
            meaningful_progress=False,
            reason="second_veto",
            metadata={"grid_changed": False},
        )
        outcome = self._dependencies.annatar(
            state,
            perception_payload,
            synthetic_execution,
            synthetic_evaluation,
            stall_reason="second_veto",
        )
        # A253: mirrors the probe-path (line ~266) / normal-cycle (line
        # ~584) call sites' own placement -- set immediately after the
        # call, before any branching on the result. Previously omitted
        # here (this call site was added in A206, the same day A205
        # established this field, and apparently never backfilled). See
        # backlog/A253.md.
        state.annatar_degraded = outcome.degraded
        if outcome.resume_mapping:
            # A241: same interception as the normal-cycle call site --
            # resume the readiness-probe loop instead of honoring what
            # would otherwise be TERMINATE. No real execute/evaluate ran
            # this cycle (the double-veto path never reaches them), so
            # current_observation is already correct as-is; returning None
            # lets the caller's own existing `continue` (right after this
            # method's call site) route back to the top of the outer
            # `while True:` loop, where the next iteration naturally
            # re-enters the readiness-gate `if` block.
            state.readiness_gate_resolved = False
            return None
        if outcome.decision == "terminate":
            return self._finish(state, WorkflowStatus.TERMINATED, "annatar_exhausted", phase_results)
        state.annatar_anchor_hint = outcome if outcome.decision in ("repeat_deepen", "repeat_retry") else None
        return None

    def _invoke_phase(self, name: str, phase_callable: Any, *args: Any) -> PhaseResult[Any]:
        result = phase_callable(*args)
        if not isinstance(result, PhaseResult):
            raise TypeError(f"{name} phase must return PhaseResult")
        return result

    @staticmethod
    def _require_payload(result: PhaseResult[Any], phase: WorkflowPhase) -> Any:
        if result.payload is None:
            raise ValueError(f"{phase.value} phase returned no payload")
        return result.payload

    @staticmethod
    def _record_execution_attempt(state: WorkflowState, execution: ExecutionResult) -> None:
        action_key = execution.candidate.book_id if execution.candidate is not None else execution.action_id
        state.action_attempt_counts[action_key] = state.action_attempt_counts.get(action_key, 0) + 1

    @staticmethod
    def _record_evaluation_state(
        state: WorkflowState,
        execution: ExecutionResult,
        evaluation: EvaluationResult,
        *,
        count_toward_no_progress: bool = True,
    ) -> None:
        action_key = execution.candidate.book_id if execution.candidate is not None else execution.action_id
        state.consecutive_no_progress_count = record_evaluation_outcome(
            no_progress_count=state.consecutive_no_progress_count,
            falsification_counts=state.action_falsification_counts,
            action_key=action_key,
            meaningful_progress=evaluation.meaningful_progress,
            falsification_delta=evaluation.falsification_delta,
            count_toward_no_progress=count_toward_no_progress,
        )
        if state.active_goal is not None:
            goal_id = state.active_goal.selected.goal_id
            if evaluation.meaningful_progress:
                state.goal_failure_counts[goal_id] = 0
            else:
                state.goal_failure_counts[goal_id] = state.goal_failure_counts.get(goal_id, 0) + 1

    @staticmethod
    def _finish(
        state: WorkflowState,
        status: WorkflowStatus,
        reason: str,
        phase_results: list[PhaseResult[Any]],
        *,
        traceback_text: str | None = None,
    ) -> WorkflowRunResult:
        state.terminated = True
        state.termination_state = status
        state.crash_traceback = traceback_text
        return WorkflowRunResult(
            status=status,
            state=state,
            phase_results=phase_results,
            reason=reason,
            traceback=traceback_text,
            completed_cycles=state.step_index,
        )
