"""Tests for A256: `perceive.py` never got the `_degraded` visibility
treatment A237/A244/A251/A253/A255 gave every other phase -- confirmed live
by a peer session (`hippocampy-75`) finding `arc_perceive_state` degrading
on every single call in a real run today (2026-09-10, task
`sb26-7fbdac44`, 4/4 calls failed) due to a real hippocampy-side `MOVED_BY`
edge-write bug. ARC_AGI's own client had zero visibility into this:
`_ingest_snapshot`'s `except Exception: return "failed"` stored the string
only in informational metadata nothing else read, and `grep -n "_degraded"
agents/arc4/perceive.py` found zero hits before this card.

This builds the `_degraded` pattern from scratch for `PerceiveAgent`,
mirroring A237/A255's established shape exactly. Fallback *behavior* is
unchanged -- `_ingest_snapshot`'s "skipped"/"failed"/"ok" return contract
and its side effects (snapshot.metadata mutation, world_model_node_writes)
are untouched; only visibility is added. See backlog/A256.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.arc4.perceive import PerceiveAgent
from agents.arc4.ports import WorkflowDependencies
from agents.arc4.telemetry import ArcV2Telemetry
from agents.arc4.types import (
    AnnatarOutcome,
    EvaluationResult,
    ExecutionResult,
    GoalHypothesis,
    PerceptionSnapshot,
    PhaseResult,
    PhaseStatus,
    ResolvedGoal,
    WorkflowDecision,
    WorkflowPhase,
    WorkflowState,
)
from agents.arc4.workflow import WorkflowLimits, WorkflowOrchestrator


# --- Shared fixtures --------------------------------------------------------


def _observation() -> dict[str, Any]:
    return {"grid": [[1, 0], [0, 1]]}


class _HealthyGraphPort:
    """The single graph_query_port call `_ingest_snapshot` touches,
    succeeding normally -- the regression baseline these tests contrast
    against."""

    def ingest_perception(self, snapshot: PerceptionSnapshot) -> dict[str, Any]:
        return {"status": "ok"}


class _RaisingGraphPort:
    """Simulates the exact live scenario A256 documents: hippocampy's
    `arc_perceive_state` genuinely raises (the `MOVED_BY` edge-write bug),
    every call."""

    def ingest_perception(self, snapshot: PerceptionSnapshot) -> dict[str, Any]:
        raise RuntimeError("hippocampy MCP not available")


# --- PerceiveAgent._ingest_snapshot's except site ---------------------------


class TestPerceiveDegradedVisibility:
    def test_raising_ingest_perception_sets_degraded_true_return_value_unchanged(self):
        agent = PerceiveAgent(graph_query_port=_RaisingGraphPort())
        state = WorkflowState()

        result = agent.perceive(state, _observation())

        assert result.payload is not None
        assert result.payload.degraded is True
        assert result.payload.metadata["graph_ingestion"] == "failed"

    def test_healthy_ingest_perception_leaves_degraded_false(self):
        agent = PerceiveAgent(graph_query_port=_HealthyGraphPort())
        state = WorkflowState()

        result = agent.perceive(state, _observation())

        assert result.payload is not None
        assert result.payload.degraded is False
        assert result.payload.metadata["graph_ingestion"] == "ok"

    def test_no_graph_port_leaves_degraded_false(self):
        """Not configured is not the same as failed -- same distinction
        every prior card in this family has emphasized."""
        agent = PerceiveAgent(graph_query_port=None)
        state = WorkflowState()

        result = agent.perceive(state, _observation())

        assert result.payload is not None
        assert result.payload.degraded is False
        assert result.payload.metadata["graph_ingestion"] == "skipped"

    def test_graph_port_without_ingest_perception_leaves_degraded_false(self):
        """A graph_query_port configured but lacking ingest_perception
        entirely -- also "skipped", not "failed"."""

        class _PortWithoutIngest:
            pass

        agent = PerceiveAgent(graph_query_port=_PortWithoutIngest())
        state = WorkflowState()

        result = agent.perceive(state, _observation())

        assert result.payload is not None
        assert result.payload.degraded is False
        assert result.payload.metadata["graph_ingestion"] == "skipped"

    def test_degraded_does_not_leak_across_successive_perceive_calls(self):
        """Same reset-per-cycle guarantee every prior card in this family
        verified."""
        agent = PerceiveAgent(graph_query_port=_RaisingGraphPort())
        state = WorkflowState()
        degraded_result = agent.perceive(state, _observation())

        agent._graph_query_port = _HealthyGraphPort()
        healthy_result = agent.perceive(WorkflowState(), _observation())

        assert degraded_result.payload.degraded is True
        assert healthy_result.payload.degraded is False


# --- PerceptionSnapshot/WorkflowState.to_dict/from_dict round-trip ---------


class TestPerceptionSnapshotDegradedRoundTrip:
    def test_defaults_false_and_survives_round_trip(self):
        snapshot = PerceptionSnapshot(observation={}, grid_hash="h1")
        assert snapshot.degraded is False

        restored = PerceptionSnapshot.from_dict(snapshot.to_dict())
        assert restored.degraded is False

    def test_true_value_survives_round_trip(self):
        snapshot = PerceptionSnapshot(observation={}, grid_hash="h1", degraded=True)
        restored = PerceptionSnapshot.from_dict(snapshot.to_dict())

        assert restored.degraded is True


class TestWorkflowStateDegradedFieldRoundTrip:
    def test_defaults_false_and_survives_round_trip(self):
        state = WorkflowState()
        assert state.perceive_degraded is False

        restored = WorkflowState.from_dict(state.to_dict())
        assert restored.perceive_degraded is False

    def test_true_value_survives_round_trip(self):
        state = WorkflowState(perceive_degraded=True)
        restored = WorkflowState.from_dict(state.to_dict())

        assert restored.perceive_degraded is True


# --- telemetry.py per-cycle summary -----------------------------------------


class TestTelemetrySurfacesPerceiveDegraded:
    def test_step_snapshot_surfaces_perceive_degraded(self):
        telemetry = ArcV2Telemetry(task_id="t1", game_id="g1")
        state = WorkflowState(perceive_degraded=True)

        snapshot = telemetry._step_snapshot((state,))

        assert snapshot["perceive_degraded"] is True

    def test_step_snapshot_defaults_false_when_state_missing_field(self):
        """getattr(..., False) degrade pattern -- must not raise/KeyError
        for a state object that predates this card (or is None)."""
        telemetry = ArcV2Telemetry(task_id="t1", game_id="g1")

        snapshot = telemetry._step_snapshot(())

        assert snapshot["perceive_degraded"] is False


# --- Integration: full workflow.py cycle with a raising graph_port ---------


def _resolve(state, perception):
    return PhaseResult(
        phase=WorkflowPhase.RESOLVE, status=PhaseStatus.OK,
        payload=ResolvedGoal(selected=GoalHypothesis(goal_id="g1", description="d", confidence=0.5)),
    )


def _plan(state, perception, goal):
    from agents.arc4.types import PlanCandidate, PlanningResult

    return PhaseResult(
        phase=WorkflowPhase.PLAN, status=PhaseStatus.OK,
        payload=PlanningResult(candidate=PlanCandidate(action_id="ACTION1", goal_id="g1", book_id="ACTION1")),
    )


def _vet(state, perception, goal, plan):
    from agents.arc4.types import VetDecision

    return PhaseResult(
        phase=WorkflowPhase.VET, status=PhaseStatus.OK,
        payload=VetDecision(approved=True, candidate=plan.candidate),
    )


def _execute(state, perception, goal, vet_decision):
    return PhaseResult(
        phase=WorkflowPhase.EXECUTE, status=PhaseStatus.OK,
        payload=ExecutionResult(
            action_id=vet_decision.candidate.action_id,
            candidate=vet_decision.candidate,
            observation={},
            did_progress=False,
        ),
    )


def _evaluate(state, perception, goal, execution):
    return PhaseResult(
        phase=WorkflowPhase.EVALUATE, status=PhaseStatus.OK,
        payload=EvaluationResult(decision=WorkflowDecision.CONTINUE, meaningful_progress=False),
    )


def _fake_annatar(state, perception, execution, evaluation, **_kwargs):
    """A250: `annatar` is a required WorkflowDependencies field now that
    it's unconditionally wired in production (since A202) -- these tests
    are about perceive degraded-visibility propagation, not Annatar's own
    decision logic, so a minimal non-terminating stand-in is enough."""
    return AnnatarOutcome(decision="advance")


def _make_dependencies(graph_query_port: Any) -> WorkflowDependencies:
    perceive_agent = PerceiveAgent(graph_query_port=graph_query_port)

    return WorkflowDependencies(
        perceive=perceive_agent.perceive,
        resolve=_resolve,
        plan=_plan,
        vet=_vet,
        execute=_execute,
        evaluate=_evaluate,
        annatar=_fake_annatar,
    )


class TestWorkflowIntegrationPerceiveDegradedPropagation:
    def test_full_cycle_with_raising_graph_port_sets_state_flag_true(self):
        deps = _make_dependencies(_RaisingGraphPort())
        orchestrator = WorkflowOrchestrator(deps, limits=WorkflowLimits(max_cycles=1))
        state = WorkflowState()

        orchestrator.run(state, _observation())

        assert state.perceive_degraded is True

    def test_full_cycle_with_healthy_graph_port_leaves_state_flag_false(self):
        deps = _make_dependencies(_HealthyGraphPort())
        orchestrator = WorkflowOrchestrator(deps, limits=WorkflowLimits(max_cycles=1))
        state = WorkflowState()

        orchestrator.run(state, _observation())

        assert state.perceive_degraded is False

    def test_full_cycle_with_no_graph_port_leaves_state_flag_false(self):
        deps = _make_dependencies(None)
        orchestrator = WorkflowOrchestrator(deps, limits=WorkflowLimits(max_cycles=1))
        state = WorkflowState()

        orchestrator.run(state, _observation())

        assert state.perceive_degraded is False
