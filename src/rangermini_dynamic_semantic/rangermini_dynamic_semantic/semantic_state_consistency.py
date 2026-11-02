"""Dependency-scoped admission of asynchronous semantic task proposals.

Optimistic concurrency and version checks are established mechanisms. This
experimental gate binds a recorded semantic request to the task and goal owner
that issued it, plus the memory facts actually supplied to the model.
"""
from dataclasses import dataclass


def _versions(values):
    if not isinstance(values, dict):
        raise ValueError("fact versions require a dictionary")
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("fact IDs require nonempty strings")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("fact versions require nonnegative integers")
    return tuple(sorted(values.items()))


@dataclass(frozen=True)
class SemanticContext:
    request_id: str
    mission_id: str
    task_revision: int
    task_index: int
    goal_id: str | None
    read_versions: tuple


class SemanticCommitGate:
    """Recorded request contexts; restarted processes reject orphan responses.

    Read sets come from the request builder, not a model's self-report. They
    must include every supplied memory fact and task prerequisite. This gate
    does not guarantee semantic correctness or replace geometry validation.
    """
    def __init__(self):
        self.issued = {}
        self.consumed = set()

    def capture(self, *, request_id, mission_id, task_revision, task_index,
                goal_id, read_fact_versions):
        if not isinstance(request_id, str) or not request_id or request_id in self.issued:
            raise ValueError("semantic request ID must be new and nonempty")
        if not isinstance(mission_id, str) or not mission_id:
            raise ValueError("mission ID required")
        if (isinstance(task_revision, bool) or not isinstance(task_revision, int)
                or task_revision < 1 or isinstance(task_index, bool)
                or not isinstance(task_index, int) or task_index < 0):
            raise ValueError("invalid task revision or index")
        if goal_id is not None and (not isinstance(goal_id, str) or not goal_id):
            raise ValueError("invalid goal ID")
        context = SemanticContext(request_id, mission_id, task_revision,
                                  task_index, goal_id, _versions(read_fact_versions))
        self.issued[request_id] = context
        return context

    def validate(self, context, *, mission_id, task_revision, task_index, goal_id,
                 current_fact_versions, update_pending=False):
        current = dict(_versions(current_fact_versions))
        if not isinstance(context, SemanticContext):
            return False, "invalid_context"
        if self.issued.get(context.request_id) != context:
            return False, "unissued_or_modified_context"
        if context.request_id in self.consumed:
            return False, "duplicate_result"
        if context.mission_id != mission_id:
            return False, "mission_changed"
        if context.task_revision != task_revision or context.task_index != task_index:
            return False, "task_changed"
        if context.goal_id != goal_id:
            return False, "goal_owner_changed"
        if update_pending:
            return False, "task_update_pending"
        for fact_id, version in context.read_versions:
            if fact_id not in current:
                return False, "dependency_missing"
            if current[fact_id] != version:
                return False, "dependency_changed"
        return True, "snapshot_valid"

    def consume(self, context):
        if self.issued.get(context.request_id) != context:
            raise ValueError("cannot consume unissued context")
        if context.request_id in self.consumed:
            raise ValueError("semantic result already consumed")
        self.consumed.add(context.request_id)
