# Execution plans

An execution plan (ExecPlan) is the versioned working record of one substantial
feature, migration or refactor. Assume the next contributor has the repository
and this plan, but no conversation history. They must be able to understand the
goal, find the relevant code, resume work and establish whether it succeeds.

## When and where

Use a plan when work spans meaningful milestones, requires investigation or
consequential decisions, changes multiple boundaries, or is likely to need a
handoff. Small, local edits can use ordinary change descriptions.

Active plans live in `docs/exec-plans/active/`; completed plans live in
`docs/exec-plans/completed/`. The index is `docs/exec-plans/README.md`.
Before starting, look for an existing plan for the task. Create one if needed,
and keep its index entry current. A roadmap describes candidate work across the
project; it is not a substitute for a task's execution state.

## Living-document rules

Write the user-visible purpose and observable acceptance first. Define unfamiliar
terms and explain the relevant modules and relationships. Use complete
repository-relative paths and exact symbols. Commands need a working directory,
prerequisites, expected observation and any meaningful side effects.

Keep the plan self-contained enough to continue. Link to supporting reference
docs, but include the critical facts and reasons rather than relying on an old
chat, a previous plan or an inaccessible external source. Useful repetition is
preferable to losing a prerequisite for safe continuation.

Update the plan after a milestone, changed approach, significant discovery or
validation result, and at every stopping point. Split partial work into what is
done and what remains. Progress should identify the immediate next action and any
blocker. Use dated checkboxes for work state and dated entries for consequential
decisions. Do not remove completed steps to make the plan look current.

Record discoveries with evidence and decisions with reasons, including rejected
alternatives when they could otherwise be reconsidered. Revise the upcoming work
to reflect those decisions. Briefly explain material plan revisions; do not leave
an obsolete approach presented as the current instruction.

Separate expected outcomes from observed results. Record actual command results,
failures, skipped checks and unavailable infrastructure. Never invent timestamps,
prior decisions or successful validation. Keep secrets out of plans and artifacts.

Plans describe authorized work; they do not expand permission to delete data,
deploy, contact others or change unrelated systems. Document partial-state and
retry behavior before consequential changes. Mark unknown recovery behavior as
unresolved rather than assuming a command is safe to repeat.

## Completion and archival

Complete a task when its agreed outcome and required acceptance are satisfied.
Record what changed, what was verified, and any remaining limitations or separately
scoped follow-up work. If required work remains, keep the plan active with its
next action or blocker; do not equate the end of a session with completion.

Promote enduring contracts and design rationale into their owning reference docs.
Move the completed plan to the completed directory and update the index and
inbound links. Preserve its progress, discoveries, decisions and evidence. The
archive complements Git and PR history; it is not redundant with them.

## Required plan structure

Use the following headings. Each section must contain task-specific information;
state when a section is not yet known or does not apply rather than inventing it.
Short plans can have short sections. Checklists belong in Progress; use prose for
explanation and code blocks for commands or concise evidence.

### Purpose / Big Picture

Explain what becomes possible for the user and how to observe it. Bound the scope
and identify explicit non-goals only when they prevent likely scope drift.

### Progress

Keep dated completed and remaining checkboxes. State current status and the next
concrete action. Retain partial completion and blockers. Example format:
`- [ ] YYYY-MM-DD HH:MMZ — Remaining action (completed portion: ...).`
Use the actual date when recording an event; the example is not an event record.

### Surprises & Discoveries

Record unexpected behavior, constraints or findings. Include the source path,
command/output or other evidence and explain the effect on the work.

### Decision Log

For consequential decisions, record the decision, rationale, meaningful rejected
alternatives, date and author. Preserve earlier decisions; label superseded ones
and explain the replacement instead of silently deleting them.

### Outcomes & Retrospective

At milestones and completion, compare results with the purpose and acceptance.
Identify remaining gaps and useful lessons. Before completion, state that the
outcome is provisional rather than implying the task is done.

### Context and Orientation

Describe the relevant existing behavior, code locations, entry points, contracts,
data/resource owners and unfamiliar terms. Include enough context for a newcomer
to locate and understand the change.

### Plan of Work

Explain the implementation sequence as bounded milestones. Tie each to an
observable result and its affected modules. Include dependencies between steps
and experiments needed to resolve uncertainty.

### Concrete Steps

List exact commands or edits with working directories, prerequisites and expected
outputs. Update the steps as the approach changes. Keep observations separate
from predictions so the next contributor knows what has actually run.

### Validation and Acceptance

Describe behavior a human can observe and the tests or demonstrations that
establish it. Include relevant failure/edge cases, expected before/after behavior,
and actual evidence. Compilation or a mock test alone may not prove the feature.

### Idempotence and Recovery

Describe repeatable operations, duplicate-write risks, partial failures, backups,
rollback or forward recovery, and how to determine the current state after an
interruption. Explain limitations rather than prescribing an unverified reset.

### Artifacts and Notes

Retain concise outputs, traces, diffs or links to artifacts that help assess and
resume the work. Include enough evidence to be useful without copying full logs.

### Interfaces and Dependencies

Name the relevant modules, functions, types, tools and external dependencies.
State the contracts or signatures needed at the end, their owners and important
constraints. Link to full definitions where appropriate.
