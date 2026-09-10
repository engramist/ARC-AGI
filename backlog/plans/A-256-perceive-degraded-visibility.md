# A256 — Perceive `degraded` Visibility: Plan

## Card metadata

- Card: `backlog/A256.md`
- Depends on: A237 (the pattern's origin), A244/A251/A253/A255 (its four prior extensions)

## Design (confirmed by direct read before writing this plan)

- `agents/arc4/perceive.py:207-227` — `_ingest_snapshot`, the sole graph-dependent method in this file; `except Exception: return "failed"` at line 217-218, no `self._degraded` anywhere in the file (`grep -n "_degraded" agents/arc4/perceive.py` → zero hits).
- `agents/arc4/perceive.py:91` — `snapshot.metadata["graph_ingestion"] = self._ingest_snapshot(snapshot, state)`, the call site inside `perceive()`. Confirm the exact surrounding code and whether `snapshot` (the `PerceptionSnapshot` being returned) is already fully constructed by this point — it almost certainly is, since `_ingest_snapshot` takes it as an argument and mutates `.metadata` in place, meaning a naive `degraded=self._degraded` at construction time would run *before* this call and be a dead write, exactly the bug A255 found and fixed for `EvaluationResult`. Confirm this ordering directly, don't assume.
- `agents/arc4/types.py` — `PerceptionSnapshot` dataclass (confirmed non-frozen, plain `@dataclass`), `EvaluationResult.degraded`/`ResolvedGoal.degraded`/`PlanningResult.degraded` as the exact shape to mirror for the new `PerceptionSnapshot.degraded` field; `WorkflowState.evaluate_degraded`/`resolve_degraded`/`annatar_degraded`/`plan_degraded`/`vet_degraded` as the shape to mirror for the new `WorkflowState.perceive_degraded` field.
- `agents/arc4/workflow.py` — confirm the exact current line for `perception = self._invoke_phase("perceive", self._dependencies.perceive, state, current_observation)` (found at `~line 123` as of A255's merge) — this is the only real perceive call site (no replan-retry duplication the way resolve/plan/vet have).
- `agents/arc4/telemetry.py` — per-cycle summary dict, where the other five `_degraded` fields are already surfaced via `bool(getattr(state, "<field>_degraded", False))`.

### The fix

**1. `perceive.py`:**

```python
class PerceiveAgent:
    def __init__(self, ...):
        ...
        self._degraded = False  # A256, mirrors PlanGenerator._degraded (A237) / GoalResolver._degraded (A251)

    def perceive(self, state: WorkflowState, observation: Mapping[str, Any]) -> PhaseResult[PerceptionSnapshot]:
        self._degraded = False  # reset at top, same convention as every other phase
        ...
        snapshot.metadata["graph_ingestion"] = self._ingest_snapshot(snapshot, state)
        # A256: _ingest_snapshot may have just set self._degraded = True on
        # failure -- snapshot was already constructed before this call, so
        # sync the instance-scratch flag onto it now, mirroring A255's own
        # necessary correction for EvaluationResult (the same
        # construct-before-the-failure-prone-call ordering applies here).
        snapshot.degraded = self._degraded
        ...
        return PhaseResult(phase=WorkflowPhase.PERCEIVE, payload=snapshot)

    def _ingest_snapshot(self, snapshot: PerceptionSnapshot, state: WorkflowState) -> str:
        ...
        try:
            result = ingest(snapshot)
        except Exception:
            self._degraded = True  # A256: was silently swallowed before
            return "failed"
        ...
```

(Illustrative — confirm the exact current `perceive()` method body/return statement shape before editing; the critical part is that `snapshot.degraded = self._degraded` must run *after* `_ingest_snapshot`'s call, not be passed into an earlier constructor call.)

**2. `types.py`:** `PerceptionSnapshot.degraded: bool = False` (wired into `to_dict()`/`from_dict()`); `WorkflowState.perceive_degraded: bool = False` (+ `to_dict`/`from_dict`).

**3. `workflow.py`:** at the one real perceive call site, immediately after `perception_payload = self._require_payload(perception, WorkflowPhase.PERCEIVE)` (or wherever the payload is extracted — confirm exact current shape), add `state.perceive_degraded = getattr(perception_payload, "degraded", False)`.

**4. `telemetry.py`:** add `"perceive_degraded": bool(getattr(state, "perceive_degraded", False))` to the per-cycle summary dict.

## Implementation approach

### Files

- Modify: `agents/arc4/perceive.py` — `__init__`, `perceive()`, `_ingest_snapshot`.
- Modify: `agents/arc4/types.py` — `PerceptionSnapshot.degraded`, `WorkflowState.perceive_degraded`.
- Modify: `agents/arc4/workflow.py` — the one perceive call site.
- Modify: `agents/arc4/telemetry.py` — per-cycle summary dict.
- Test: new `tests/test_a256_perceive_degraded_visibility.py`.

### TDD

- New test: a fake `graph_query_port` whose `ingest_perception` raises → `perceive()` completes without propagating, returns a `PerceptionSnapshot` with `degraded=True` and `metadata["graph_ingestion"] == "failed"`. Confirm this test fails against the current pre-fix code first (proving the gap is real), then implement.
- New test: a healthy (non-raising) `ingest_perception` → `degraded=False`.
- New test: `graph_query_port=None` → `degraded=False` — the "not configured ≠ failed" distinction.
- New test: `WorkflowState.perceive_degraded` correctly reflects the perceive call's `degraded` value after a real `workflow.py` cycle (an integration-level test, mirroring A244/A255's own `TestWorkflowIntegration*DegradedPropagation` style).
- New test: `telemetry.py`'s per-cycle summary surfaces `perceive_degraded` correctly, defaulting `False` when `state` is missing the field or `state` is `None`.
- Regression: existing perceive-adjacent tests (entity classification, loop-signal detection, etc.) continue to pass unchanged.

### Validation commands

```bash
.venv/bin/python -m pytest tests/test_a256_perceive_degraded_visibility.py -v
.venv/bin/python -m pytest -k perceive
make test-a
make test-all
```

### Live-verify

Same environment/discipline as every prior card this investigation. Given the motivating hippocampy bug (`MOVED_BY` edge handler) is confirmed still open as of this card's filing, a live smoke run against the puzzle types that exercise entity movement has a real chance of showing `perceive_degraded=True` for a genuine reason — report honestly whichever is actually observed (a real degraded reading, or a clean healthy run if the bug happens to not trigger, or if their B421 fix has landed by the time this card is implemented). The TDD suite is the primary evidence for the new-behavior claim regardless.

## Assumptions/defaults

- Exact mirror of A237/A255's established shape, applied to a phase that previously had none of this infrastructure — the first "from scratch" build in this family rather than an extension of existing per-file machinery.
- If `PerceptionSnapshot` turns out to be constructed and returned in a way that makes the post-hoc `snapshot.degraded = self._degraded` assignment awkward (e.g., an intervening return path), investigate the real control flow directly rather than forcing the sketch above — the ordering constraint (sync after `_ingest_snapshot` runs, not before) is the hard requirement, the exact mechanics are flexible.
