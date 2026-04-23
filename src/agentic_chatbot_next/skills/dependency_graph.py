from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from agentic_chatbot_next.persistence.postgres.skills import SkillPackRecord


def _family_id(record: SkillPackRecord) -> str:
    return str(record.version_parent or record.skill_id or "").strip()


def _coerce_depends_on_skills(raw: Any) -> List[str]:
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Sequence):
        values = [str(item) for item in raw]
    else:
        values = []
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def dependency_families_for_record(record: SkillPackRecord) -> List[str]:
    return _coerce_depends_on_skills(dict(record.controller_hints or {}).get("depends_on_skills"))


def _timestamp_key(value: str) -> tuple[int, str]:
    text = str(value or "").strip()
    if not text:
        return (0, "")
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return (0, text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (1, parsed.astimezone(timezone.utc).isoformat())


def _record_sort_key(record: SkillPackRecord) -> tuple[tuple[int, str], str]:
    return (_timestamp_key(record.updated_at), str(record.skill_id or ""))


def _cycle_signature(cycle: Sequence[str]) -> tuple[str, ...]:
    nodes = [str(item) for item in cycle if str(item)]
    if not nodes:
        return ()
    if len(nodes) > 1 and nodes[0] == nodes[-1]:
        nodes = nodes[:-1]
    if not nodes:
        return ()
    rotations: List[tuple[str, ...]] = []
    for index in range(len(nodes)):
        rotated = tuple(nodes[index:] + nodes[:index])
        rotations.append(rotated)
        rotations.append(tuple(reversed(rotated)))
    return min(rotations)


@dataclass
class BlockedDependent:
    skill_id: str
    skill_family_id: str
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_family_id": self.skill_family_id,
            "name": self.name,
        }


@dataclass
class SkillDependencyValidation:
    skill_id: str = ""
    skill_family_id: str = ""
    evaluation_mode: str = "current_active_graph"
    depends_on_skills: List[str] = field(default_factory=list)
    resolved_dependency_skill_ids: Dict[str, str] = field(default_factory=dict)
    missing_dependencies: List[str] = field(default_factory=list)
    cycles: List[List[str]] = field(default_factory=list)
    blocked_dependents: List[BlockedDependent] = field(default_factory=list)
    impacted_families: List[str] = field(default_factory=list)
    dependency_state: str = "healthy"
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_family_id": self.skill_family_id,
            "evaluation_mode": self.evaluation_mode,
            "depends_on_skills": list(self.depends_on_skills),
            "resolved_dependency_skill_ids": dict(self.resolved_dependency_skill_ids),
            "missing_dependencies": list(self.missing_dependencies),
            "cycles": [list(cycle) for cycle in self.cycles],
            "blocked_dependents": [item.to_dict() for item in self.blocked_dependents],
            "impacted_families": list(self.impacted_families),
            "dependency_state": self.dependency_state,
            "is_valid": bool(self.is_valid),
            "warnings": list(self.warnings),
        }


@dataclass
class SkillDependencyGraphSummary:
    active_family_count: int = 0
    invalid_family_count: int = 0
    missing_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    cycles: List[List[str]] = field(default_factory=list)
    blocked_families: Dict[str, List[str]] = field(default_factory=dict)
    valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_family_count": int(self.active_family_count),
            "invalid_family_count": int(self.invalid_family_count),
            "missing_dependencies": {
                str(key): list(value)
                for key, value in dict(self.missing_dependencies).items()
            },
            "cycles": [list(cycle) for cycle in self.cycles],
            "blocked_families": {
                str(key): list(value)
                for key, value in dict(self.blocked_families).items()
            },
            "valid": bool(self.valid),
        }


@dataclass
class SkillDependencyGraph:
    records_by_id: Dict[str, SkillPackRecord]
    records_by_family: Dict[str, List[SkillPackRecord]]
    active_records_by_family: Dict[str, SkillPackRecord]
    dependencies_by_family: Dict[str, List[str]]
    reverse_dependencies_by_family: Dict[str, List[str]]
    missing_dependencies_by_family: Dict[str, List[str]]
    cycles: List[List[str]]
    invalid_families: Dict[str, List[str]]
    cycle_families: Dict[str, List[List[str]]] = field(default_factory=dict)

    @classmethod
    def from_records(cls, records: Iterable[SkillPackRecord]) -> "SkillDependencyGraph":
        records_by_id: Dict[str, SkillPackRecord] = {}
        records_by_family: Dict[str, List[SkillPackRecord]] = {}
        active_records_by_family: Dict[str, SkillPackRecord] = {}
        dependencies_by_family: Dict[str, List[str]] = {}
        reverse_dependencies_by_family: Dict[str, set[str]] = {}
        missing_dependencies_by_family: Dict[str, List[str]] = {}
        cycle_families: Dict[str, List[List[str]]] = {}

        for record in records:
            records_by_id[record.skill_id] = record
            family_id = _family_id(record)
            records_by_family.setdefault(family_id, []).append(record)

        for family_id, family_records in records_by_family.items():
            active_records = [
                record
                for record in family_records
                if bool(record.enabled) and str(record.status or "").strip().lower() == "active"
            ]
            if active_records:
                active_records_by_family[family_id] = max(active_records, key=_record_sort_key)

        for family_id, record in active_records_by_family.items():
            dependencies = dependency_families_for_record(record)
            dependencies_by_family[family_id] = dependencies
            for dependency in dependencies:
                reverse_dependencies_by_family.setdefault(dependency, set()).add(family_id)
            missing = [
                dependency
                for dependency in dependencies
                if dependency not in active_records_by_family
            ]
            if missing:
                missing_dependencies_by_family[family_id] = missing

        cycles = _find_cycles(dependencies_by_family, active_records_by_family.keys())
        for cycle in cycles:
            for family_id in cycle:
                cycle_families.setdefault(family_id, []).append(list(cycle))

        invalid_families: Dict[str, List[str]] = {}
        queue: List[str] = []

        for family_id, missing in missing_dependencies_by_family.items():
            invalid_families[family_id] = [f"missing:{dependency}" for dependency in missing]
            queue.append(family_id)
        for family_id, family_cycles in cycle_families.items():
            reasons = invalid_families.setdefault(family_id, [])
            reasons.extend(
                f"cycle:{' -> '.join(cycle + [cycle[0]])}"
                for cycle in family_cycles
                if cycle
            )
            queue.append(family_id)

        while queue:
            blocked_family = queue.pop(0)
            for dependent_family in sorted(reverse_dependencies_by_family.get(blocked_family, set())):
                reasons = invalid_families.setdefault(dependent_family, [])
                marker = f"blocked:{blocked_family}"
                if marker in reasons:
                    continue
                reasons.append(marker)
                queue.append(dependent_family)

        return cls(
            records_by_id=records_by_id,
            records_by_family=records_by_family,
            active_records_by_family=active_records_by_family,
            dependencies_by_family=dependencies_by_family,
            reverse_dependencies_by_family={
                key: sorted(value) for key, value in reverse_dependencies_by_family.items()
            },
            missing_dependencies_by_family=missing_dependencies_by_family,
            cycles=cycles,
            invalid_families=invalid_families,
            cycle_families=cycle_families,
        )

    def summary(self) -> SkillDependencyGraphSummary:
        return SkillDependencyGraphSummary(
            active_family_count=len(self.active_records_by_family),
            invalid_family_count=len(self.invalid_families),
            missing_dependencies={
                str(key): list(value)
                for key, value in self.missing_dependencies_by_family.items()
            },
            cycles=[list(cycle) for cycle in self.cycles],
            blocked_families={
                str(key): list(value)
                for key, value in self.invalid_families.items()
            },
            valid=not bool(self.invalid_families),
        )

    def active_record_for_identifier(self, identifier: str) -> SkillPackRecord | None:
        clean = str(identifier or "").strip()
        if not clean:
            return None
        if clean in self.active_records_by_family:
            return self.active_records_by_family.get(clean)
        record = self.records_by_id.get(clean)
        if record is None:
            return None
        return self.active_records_by_family.get(_family_id(record))

    def valid_active_skill_ids(self) -> set[str]:
        return {
            record.skill_id
            for family_id, record in self.active_records_by_family.items()
            if family_id not in self.invalid_families
        }

    def dependency_validation_for_skill(
        self,
        skill_id: str,
        *,
        evaluation_mode: str = "current_active_graph",
    ) -> SkillDependencyValidation:
        record = self.records_by_id.get(str(skill_id or "").strip())
        if record is None:
            return SkillDependencyValidation(
                skill_id=str(skill_id or ""),
                skill_family_id="",
                evaluation_mode=evaluation_mode,
                dependency_state="blocked",
                is_valid=False,
                warnings=["Skill record was not found."],
            )
        family_id = _family_id(record)
        dependencies = dependency_families_for_record(record)
        resolved_dependency_skill_ids: Dict[str, str] = {}
        for dependency in dependencies:
            active = self.active_records_by_family.get(dependency)
            if active is not None:
                resolved_dependency_skill_ids[dependency] = active.skill_id

        missing_dependencies = list(self.missing_dependencies_by_family.get(family_id, []))
        cycles = [list(cycle) for cycle in self.cycle_families.get(family_id, [])]
        invalid_reasons = list(self.invalid_families.get(family_id, []))
        impacted_families: List[str] = []
        blocked_dependents: List[BlockedDependent] = []

        dependency_state = "healthy"
        if invalid_reasons:
            dependency_state = "blocked" if evaluation_mode == "current_active_graph" else "warning"
            impacted_families = sorted(
                _transitive_dependents(self.reverse_dependencies_by_family, [family_id])
            )
            blocked_dependents = [
                BlockedDependent(
                    skill_id=dependent_record.skill_id,
                    skill_family_id=dependent_family,
                    name=str(dependent_record.name or dependent_family),
                )
                for dependent_family in impacted_families
                if dependent_family in self.active_records_by_family
                for dependent_record in [self.active_records_by_family[dependent_family]]
            ]

        warnings: List[str] = []
        for reason in invalid_reasons:
            if reason.startswith("blocked:"):
                warnings.append(
                    f"Blocked because dependency family '{reason.split(':', 1)[1]}' is invalid."
                )

        return SkillDependencyValidation(
            skill_id=record.skill_id,
            skill_family_id=family_id,
            evaluation_mode=evaluation_mode,
            depends_on_skills=dependencies,
            resolved_dependency_skill_ids=resolved_dependency_skill_ids,
            missing_dependencies=missing_dependencies,
            cycles=cycles,
            blocked_dependents=blocked_dependents,
            impacted_families=impacted_families,
            dependency_state=dependency_state,
            is_valid=not bool(invalid_reasons),
            warnings=warnings,
        )


def _find_cycles(
    dependencies_by_family: Mapping[str, Sequence[str]],
    families: Iterable[str],
) -> List[List[str]]:
    family_set = {str(item) for item in families if str(item)}
    cycles: Dict[tuple[str, ...], List[str]] = {}
    visited: set[str] = set()
    stack: List[str] = []
    in_stack: set[str] = set()

    def visit(family_id: str) -> None:
        if family_id in visited:
            return
        visited.add(family_id)
        stack.append(family_id)
        in_stack.add(family_id)
        for dependency in dependencies_by_family.get(family_id, []):
            if dependency not in family_set:
                continue
            if dependency in in_stack:
                try:
                    start = stack.index(dependency)
                except ValueError:
                    continue
                cycle = stack[start:] + [dependency]
                signature = _cycle_signature(cycle)
                if signature:
                    cycles.setdefault(signature, list(signature))
                continue
            visit(dependency)
        in_stack.discard(family_id)
        stack.pop()

    for family_id in sorted(family_set):
        visit(family_id)
    return [list(cycle) for cycle in cycles.values()]


def _transitive_dependents(
    reverse_dependencies_by_family: Mapping[str, Sequence[str]],
    roots: Iterable[str],
) -> set[str]:
    pending = [str(item) for item in roots if str(item)]
    seen: set[str] = set()
    result: set[str] = set()
    while pending:
        family_id = pending.pop(0)
        if family_id in seen:
            continue
        seen.add(family_id)
        for dependent in reverse_dependencies_by_family.get(family_id, []):
            clean = str(dependent or "").strip()
            if not clean or clean in result:
                continue
            result.add(clean)
            pending.append(clean)
    return result


def build_skill_dependency_graph(records: Iterable[SkillPackRecord]) -> SkillDependencyGraph:
    return SkillDependencyGraph.from_records(records)


def build_record_activation_validation(
    records: Iterable[SkillPackRecord],
    *,
    skill_id: str,
) -> SkillDependencyValidation:
    simulated: List[SkillPackRecord] = []
    for record in records:
        if record.skill_id == skill_id:
            simulated.append(replace(record, enabled=True, status="active"))
        else:
            simulated.append(replace(record))
    graph = build_skill_dependency_graph(simulated)
    return graph.dependency_validation_for_skill(
        skill_id,
        evaluation_mode="activation_preview",
    )


def build_transition_validation(
    records: Iterable[SkillPackRecord],
    *,
    overrides: Mapping[str, Mapping[str, Any]],
    primary_skill_id: str,
    action: str,
) -> SkillDependencyValidation:
    simulated: List[SkillPackRecord] = []
    for record in records:
        update = dict(overrides.get(record.skill_id) or {})
        simulated.append(
            replace(
                record,
                enabled=bool(update.get("enabled", record.enabled)),
                status=str(update.get("status", record.status)),
            )
        )
    graph = build_skill_dependency_graph(simulated)
    validation = graph.dependency_validation_for_skill(
        primary_skill_id,
        evaluation_mode=f"{action}_preview",
    )
    current_graph = build_skill_dependency_graph(records)
    primary_record = current_graph.records_by_id.get(primary_skill_id)
    if primary_record is not None:
        family_id = _family_id(primary_record)
        current_active = family_id in current_graph.active_records_by_family
        next_active = family_id in graph.active_records_by_family
        if current_active and not next_active:
            impacted = sorted(
                _transitive_dependents(current_graph.reverse_dependencies_by_family, [family_id])
            )
            validation.impacted_families = impacted
            validation.blocked_dependents = [
                BlockedDependent(
                    skill_id=dependent_record.skill_id,
                    skill_family_id=dependent_family,
                    name=str(dependent_record.name or dependent_family),
                )
                for dependent_family in impacted
                if dependent_family in current_graph.active_records_by_family
                for dependent_record in [current_graph.active_records_by_family[dependent_family]]
            ]
            if validation.blocked_dependents:
                validation.dependency_state = "blocked"
                validation.is_valid = False
                validation.warnings.append(
                    "This transition would break active dependent skill families."
                )
    return validation


def build_dependency_error_payload(
    *,
    message: str,
    action: str,
    validation: SkillDependencyValidation,
) -> Dict[str, Any]:
    return {
        "message": message,
        "action": action,
        "skill_id": validation.skill_id,
        "skill_family_id": validation.skill_family_id,
        "dependency_validation": validation.to_dict(),
    }


__all__ = [
    "BlockedDependent",
    "SkillDependencyGraph",
    "SkillDependencyGraphSummary",
    "SkillDependencyValidation",
    "build_dependency_error_payload",
    "build_record_activation_validation",
    "build_skill_dependency_graph",
    "build_transition_validation",
    "dependency_families_for_record",
]
