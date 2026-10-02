---
title: "TASK: {{TASK_TITLE}}"
type: task
status: backlog
timebox: 30m
priority: P1
tags:
  - second-brain/task
  - task/gsd
---

# 🎯 Task: {{TASK_TITLE}}

### 1. Goal (Atomic & Measurable)
- {{TASK_GOAL}}

### 2. Constraints & Time-Box
- **Estimated Duration**: `{{TIMEBOX}}` (Max 45m; break into sub-tasks if larger)
- **Target Files / Scope**:
  - `{{TARGET_FILES}}`

### 3. Action Checklist (Max 3–5 Discrete Actions)
- [ ] 1. {{STEP_1}}
- [ ] 2. {{STEP_2}}
- [ ] 3. {{STEP_3}}

### 4. Definition of Done (Binary Verifiable Gates)
- [ ] Automated verification passes (e.g. `python scripts/vault_linter.py vault_template`)
- [ ] Target artifact created or updated with 0 broken links and 0 AI slop phrases.

### 5. Failure Rollback Plan
- **Rollback Command**: `git checkout -- {{TARGET_FILES}}`
