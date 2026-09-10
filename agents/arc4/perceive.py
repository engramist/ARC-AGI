"""Deterministic ARC v2 perception module."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from typing import Any, Mapping, Sequence

from .ports import GraphQueryPort
from .rule_extraction import classify_effect_type
from .types import PerceivedEntity, PerceptionSnapshot, PhaseResult, WorkflowPhase, WorkflowState


class PerceiveAgent:
    """Build a stable perception snapshot from a raw observation."""

    def __init__(self, graph_query_port: GraphQueryPort | None = None, *, loop_window: int = 32) -> None:
        self._graph_query_port = graph_query_port
        self._loop_window = max(1, int(loop_window))
        # A256: mirrors PlanGenerator._degraded (A237) / GoalResolver
        # ._degraded (A251) -- instance-scratch flag surfacing whether
        # this file's sole graph-dependent call (_ingest_snapshot) raised
        # during the most recent perceive() call.
        self._degraded = False

    def __call__(self, state: WorkflowState, observation: Mapping[str, Any]) -> PhaseResult[PerceptionSnapshot]:
        return self.perceive(state, observation)

    def perceive(self, state: WorkflowState, observation: Mapping[str, Any]) -> PhaseResult[PerceptionSnapshot]:
        self._degraded = False  # A256: reset at top, same convention as every other phase
        normalized_observation = self._normalize_observation(observation)
        grid = self._extract_grid(normalized_observation)
        normalized_grid = self._normalize_grid(grid)
        grid_hash = self._hash_grid(normalized_grid)
        grid_shape = self._grid_shape(normalized_grid)
        raw_entities = self._extract_entities(normalized_grid)
        previous_entities = state.previous_entities or ()
        entities = self._assign_correspondence(state, raw_entities)
        # A219: disappearance detection and per-entity effect-type
        # classification, computed here (not in `_assign_correspondence`
        # itself) while both the old `previous_entities` and the new
        # `entities` are both in hand -- see `_find_disappeared_entities`
        # and `_compute_entity_effects` docstrings for why this is a
        # sibling computation rather than a change to
        # `_assign_correspondence`'s own return contract.
        disappeared_entities = self._find_disappeared_entities(previous_entities, entities)
        entity_effects = self._compute_entity_effects(previous_entities, entities, disappeared_entities)
        grid_text = self._encode_grid_text(normalized_grid)
        grid_diff = self._diff_grids(state.previous_grid, normalized_grid)

        state.previous_grid_hash = grid_hash
        state.previous_grid = normalized_grid
        state.previous_entities = entities
        state.loop_history.append(grid_hash)
        state.loop_history_pointer = len(state.loop_history) - 1
        if len(state.loop_history) > self._loop_window:
            overflow = len(state.loop_history) - self._loop_window
            del state.loop_history[:overflow]
            state.loop_history_pointer = len(state.loop_history) - 1

        repeated_grid_count = sum(1 for known_hash in state.loop_history if known_hash == grid_hash)
        loop_signal = repeated_grid_count > 1

        snapshot = PerceptionSnapshot(
            observation=normalized_observation,
            grid_hash=grid_hash,
            grid_shape=grid_shape,
            loop_signal=loop_signal,
            repeated_grid_count=repeated_grid_count,
            entities=entities,
            metadata={
                "entity_count": len(entities),
                "observation_keys": tuple(sorted(normalized_observation.keys())),
                "grid_source": self._grid_source_key(normalized_observation),
                "grid_text": grid_text,
                "grid_diff": grid_diff,
                # A219: telemetry-only entity-level effect classification
                # (translation/growth/shrink/appearance/disappearance/
                # unchanged) -- not consumed by scoring or graph writes in
                # this card, see backlog/A219.md.
                "entity_effects": entity_effects,
                # A221 Finding 2: disappearance is a genuinely new causal
                # fact (A175's correspondence tracking had no disappearance
                # detection before A219) -- unlike the other five effect
                # types, this one is promoted to a real graph write
                # (ingest_perception, see graph_queries.py). Plain-dict
                # shape (PerceivedEntity.to_dict()), matching how `entities`
                # itself is handled in PerceptionSnapshot.to_dict() -- not
                # raw PerceivedEntity instances, which would break
                # metadata's direct pass-through there.
                "disappeared_entities": [e.to_dict() for e in disappeared_entities],
            },
        )

        snapshot.metadata["graph_ingestion"] = self._ingest_snapshot(snapshot, state)
        # A256: _ingest_snapshot may have just set self._degraded = True on
        # failure -- snapshot was already constructed above (before this
        # call), so sync the instance-scratch flag onto it now, mirroring
        # A255's own necessary correction for EvaluationResult (the same
        # construct-before-the-failure-prone-call ordering applies here).
        snapshot.degraded = self._degraded
        return PhaseResult(phase=WorkflowPhase.PERCEIVE, payload=snapshot)

    @staticmethod
    def _assign_correspondence(
        state: WorkflowState,
        entities: tuple[PerceivedEntity, ...],
        *,
        radius: float = 6.0,
    ) -> tuple[PerceivedEntity, ...]:
        """A175: frame-to-frame entity correspondence via bounded-radius nearest-
        centroid matching (same color). Every entity previously collapsed to one
        degenerate graph node because the server keys identity on fields the
        client never sent -- this assigns a stable `entity_ref` that survives
        across steps for the same physical object, which the client can now
        actually send. Simple greedy matching, not optimal bipartite assignment
        (see backlog/A175.md Step 0) -- start simple, escalate only if live
        evidence shows systematic misassignment.
        """
        previous = state.previous_entities or ()
        claimed: set[int] = set()
        updated: list[PerceivedEntity] = []
        for entity in entities:
            centroid = entity.attributes.get("centroid")
            best_index: int | None = None
            best_distance: float | None = None
            if centroid is not None:
                for index, prev_entity in enumerate(previous):
                    if index in claimed or prev_entity.value != entity.value:
                        continue
                    prev_centroid = prev_entity.attributes.get("centroid")
                    if prev_centroid is None:
                        continue
                    distance = math.dist(centroid, prev_centroid)
                    if distance <= radius and (best_distance is None or distance < best_distance):
                        best_distance = distance
                        best_index = index
            if best_index is not None:
                claimed.add(best_index)
                entity_ref = previous[best_index].attributes.get("entity_ref")
            else:
                entity_ref = state.next_entity_ref
                state.next_entity_ref += 1
            new_attributes = dict(entity.attributes)
            new_attributes["entity_ref"] = entity_ref
            updated.append(PerceivedEntity(kind=entity.kind, value=entity.value, attributes=new_attributes))
        return tuple(updated)

    @staticmethod
    def _find_disappeared_entities(
        previous: tuple[PerceivedEntity, ...],
        current: tuple[PerceivedEntity, ...],
    ) -> tuple[PerceivedEntity, ...]:
        """A219: which of the *previous* frame's entities went unclaimed this
        frame -- not computed anywhere before this card (A216 Part 2's gap:
        "there's no explicit 'this entity went unclaimed' pass").

        Added as a sibling that post-hoc diffs `entity_ref` sets, rather than
        changing `_assign_correspondence`'s own return contract or internals:
        `_assign_correspondence` has exactly one call site (`perceive()`,
        directly above), so changing its signature was a viable option too,
        but this wrapper approach is strictly lower-risk (zero chance of an
        accidental behavior change to the existing greedy-matching logic,
        which several other tests pin byte-for-byte) and is just as correct.
        It relies on an invariant `_assign_correspondence` already guarantees:
        `entity_ref` values are unique both within one frame's entities and
        across frames (each previous entity is claimed by at most one current
        entity via its internal `claimed` set, and freshly-minted refs come
        from a monotonic `state.next_entity_ref` counter that never repeats
        an in-use value) -- so a previous entity is "unclaimed" this frame
        iff its `entity_ref` does not appear among the current frame's
        `entity_ref` values.
        """
        current_refs = {entity.attributes.get("entity_ref") for entity in current}
        return tuple(entity for entity in previous if entity.attributes.get("entity_ref") not in current_refs)

    @staticmethod
    def _compute_entity_effects(
        previous: tuple[PerceivedEntity, ...],
        current: tuple[PerceivedEntity, ...],
        disappeared: tuple[PerceivedEntity, ...],
    ) -> list[dict[str, Any]]:
        """A219: classify every entity touched this frame -- matched/new
        entities in `current`, plus `disappeared` entities from
        `_find_disappeared_entities` -- via `classify_effect_type()`, and
        return a plain-dict list suitable for telemetry (see
        `telemetry.py::_step_snapshot`, which mirrors this into the step
        trace). Read-only: nothing here is consumed by scoring or graph
        writes in this card.
        """
        previous_by_ref = {entity.attributes.get("entity_ref"): entity.attributes for entity in previous}
        effects: list[dict[str, Any]] = []
        for entity in current:
            entity_ref = entity.attributes.get("entity_ref")
            previous_attributes = previous_by_ref.get(entity_ref)
            effect_type = classify_effect_type(previous_attributes, entity.attributes)
            effects.append(
                {
                    "entity_ref": entity_ref,
                    "kind": entity.kind,
                    "value": entity.value,
                    "effect_type": effect_type.value,
                }
            )
        for entity in disappeared:
            effect_type = classify_effect_type(entity.attributes, None)
            effects.append(
                {
                    "entity_ref": entity.attributes.get("entity_ref"),
                    "kind": entity.kind,
                    "value": entity.value,
                    "effect_type": effect_type.value,
                }
            )
        return effects

    def _ingest_snapshot(self, snapshot: PerceptionSnapshot, state: WorkflowState) -> str:
        if self._graph_query_port is None:
            return "skipped"

        ingest = getattr(self._graph_query_port, "ingest_perception", None)
        if ingest is None:
            return "skipped"

        try:
            result = ingest(snapshot)
        except Exception:
            self._degraded = True  # A256: was silently swallowed before
            return "failed"

        if result is None:
            # A183: no result to inspect -- can't confirm a real write
            # happened, so don't count it toward world_model_node_writes.
            return "ok"
        snapshot.metadata["graph_ingestion_result"] = result
        if isinstance(result, Mapping) and result.get("status") == "ok":
            state.world_model_node_writes += 1
        return "ok"

    @staticmethod
    def _normalize_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
        return dict(observation)

    def _extract_grid(self, observation: Mapping[str, Any]) -> list[list[Any]] | None:
        grid_like = self._find_grid_like(observation)
        if grid_like is None:
            return None
        return [list(row) for row in grid_like]

    def _find_grid_like(self, value: Any, _seen: set[int] | None = None) -> Sequence[Sequence[Any]] | None:
        if _seen is None:
            _seen = set()
        if isinstance(value, Mapping):
            marker = id(value)
            if marker in _seen:
                return None
            _seen.add(marker)
            for key in ("grid", "board", "cells"):
                candidate = value.get(key)
                if self._is_grid(candidate):
                    return candidate  # type: ignore[return-value]
            for key in ("state", "observation", "payload", "game_state"):
                nested = value.get(key)
                found = self._find_grid_like(nested, _seen)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _is_grid(value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and bool(value) and all(
            isinstance(row, Sequence) and not isinstance(row, (str, bytes)) for row in value
        )

    @staticmethod
    def _normalize_grid(grid: Sequence[Sequence[Any]] | None) -> list[list[Any]]:
        if grid is None:
            return []
        return [list(row) for row in grid]

    @staticmethod
    def _hash_grid(grid: Sequence[Sequence[Any]]) -> str:
        payload = json.dumps(grid, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _grid_shape(grid: Sequence[Sequence[Any]]) -> tuple[int, int] | None:
        if not grid:
            return None
        row_count = len(grid)
        col_count = max((len(row) for row in grid), default=0)
        return row_count, col_count

    @staticmethod
    def _encode_grid_text(grid: Sequence[Sequence[Any]], *, max_cells: int = 4096) -> str:
        """A169: compact one-char-per-cell text encoding of the grid for LLM prompts.

        Previously the LLM never saw the grid at all, only a hash plus
        abstracted blob statistics. ARC's palette is single digits, so a
        digit-per-cell, newline-per-row encoding is natural and compact.
        """
        if not grid:
            return ""
        rows = len(grid)
        cols = max((len(row) for row in grid), default=0)
        if rows * cols > max_cells:
            return f"grid omitted: {rows}x{cols} exceeds {max_cells}-cell encoding limit"
        return "\n".join("".join(str(cell) for cell in row) for row in grid)

    @staticmethod
    def _diff_grids(
        previous: Sequence[Sequence[Any]] | None,
        current: Sequence[Sequence[Any]],
        *,
        max_entries: int = 50,
    ) -> dict[str, Any]:
        """A170: structured before/after cell diff, since every action's effect
        previously collapsed to a boolean grid_changed -- no way to represent
        "clicking here turned 3 cells from color 2 to 5", which is exactly the
        evidence shape ARC-style causal reasoning needs.
        """
        if previous is None or len(previous) != len(current):
            return {"changed_cells": [], "changed_count": 0, "truncated": False}
        changes: list[dict[str, Any]] = []
        for row_index, (prev_row, cur_row) in enumerate(zip(previous, current)):
            if len(prev_row) != len(cur_row):
                continue
            for col_index, (prev_val, cur_val) in enumerate(zip(prev_row, cur_row)):
                if prev_val != cur_val:
                    changes.append({"row": row_index, "col": col_index, "from": prev_val, "to": cur_val})
        return {
            "changed_cells": changes[:max_entries],
            "changed_count": len(changes),
            "truncated": len(changes) > max_entries,
        }

    def _extract_entities(self, grid: Sequence[Sequence[Any]]) -> tuple[PerceivedEntity, ...]:
        if not grid:
            return ()

        rows = len(grid)
        cols = max((len(row) for row in grid), default=0)
        visited: set[tuple[int, int]] = set()
        entities: list[PerceivedEntity] = []

        for row_index in range(rows):
            for col_index in range(len(grid[row_index])):
                if (row_index, col_index) in visited:
                    continue
                value = grid[row_index][col_index]
                if value in (0, None):
                    continue
                component = self._collect_component(grid, row_index, col_index, visited)
                if not component:
                    continue
                entities.append(self._component_to_entity(grid, component, rows, cols))

        return tuple(entities)

    def _collect_component(
        self,
        grid: Sequence[Sequence[Any]],
        start_row: int,
        start_col: int,
        visited: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        target = grid[start_row][start_col]
        queue: deque[tuple[int, int]] = deque([(start_row, start_col)])
        visited.add((start_row, start_col))
        component: list[tuple[int, int]] = []

        while queue:
            row_index, col_index = queue.popleft()
            component.append((row_index, col_index))
            for next_row, next_col in (
                (row_index - 1, col_index),
                (row_index + 1, col_index),
                (row_index, col_index - 1),
                (row_index, col_index + 1),
            ):
                if next_row < 0 or next_row >= len(grid):
                    continue
                if next_col < 0 or next_col >= len(grid[next_row]):
                    continue
                if (next_row, next_col) in visited:
                    continue
                if grid[next_row][next_col] != target:
                    continue
                visited.add((next_row, next_col))
                queue.append((next_row, next_col))

        return component

    def _component_to_entity(
        self,
        grid: Sequence[Sequence[Any]],
        component: Sequence[tuple[int, int]],
        total_rows: int,
        total_cols: int,
    ) -> PerceivedEntity:
        row_values = [row for row, _ in component]
        col_values = [col for _, col in component]
        min_row = min(row_values)
        max_row = max(row_values)
        min_col = min(col_values)
        max_col = max(col_values)
        cell_count = len(component)
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        centroid_row = sum(row_values) / cell_count
        centroid_col = sum(col_values) / cell_count
        color = grid[component[0][0]][component[0][1]]
        aspect_ratio = round(width / max(height, 1), 3)
        fill_ratio = round(cell_count / max(width * height, 1), 3)

        if cell_count == 1:
            shape = "point"
        elif width == 1 or height == 1:
            shape = "line"
        elif 0.75 <= aspect_ratio <= 1.25:
            shape = "block"
        else:
            shape = "blob"

        return PerceivedEntity(
            kind=shape,
            value=str(color),
            attributes={
                "color": color,
                "cell_count": cell_count,
                "bbox": (min_row, min_col, max_row, max_col),
                "centroid": (round(centroid_row, 3), round(centroid_col, 3)),
                "width": width,
                "height": height,
                "fill_ratio": fill_ratio,
                "aspect_ratio": aspect_ratio,
                "coverage": round(cell_count / max(total_rows * total_cols, 1), 3),
            },
        )

    @staticmethod
    def _grid_source_key(observation: Mapping[str, Any]) -> str:
        for key in ("grid", "board", "cells"):
            if key in observation:
                return key
        for key in ("state", "observation", "payload", "game_state"):
            nested = observation.get(key)
            if isinstance(nested, Mapping):
                for nested_key in ("grid", "board", "cells"):
                    if nested_key in nested:
                        return f"{key}.{nested_key}"
        return "unknown"