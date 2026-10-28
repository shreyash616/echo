---
name: gemini-handoff
description: Hand one tiny, fully specified echo task to the Gemini agent (agy CLI) when Claude's 5-hour or weekly quota is close to its limit. Not for normal work.
---

# Gemini hand-off

Use only when Claude's quota is nearly exhausted and work would otherwise stop. Claude stays the primary agent and reviews everything Gemini produces.

## Cutting the task
Gemini is weak at coding, so the spec must leave nothing to judgement:
- **One change, normally one file.** List every file it may edit in `files`. The wrapper reverts edits to any other file.
- **Literal steps.** Name the function and line, and quote the exact code to replace and the replacement. Avoid "refactor" or "improve".
- **Claude chooses the verification.** List commands in `verify`. The wrapper runs them; Gemini runs nothing.
- **Inside the current packet.** The files must be in the current FIX_PLAN_2.md section 5 packet, and the task must not edit a check to make it pass.

Write the spec to `.claude/gemini/tasks/<id>.json`:
```json
{"id": "p2-sat-01", "goal": "one sentence",
 "files": ["backend/app/audio/features.py"],
 "steps": ["In `saturation()` at features.py:427 replace `...` with `...`"],
 "verify": ["py -m pytest -q tests/test_features_golden.py"]}
```

## Running it
```bash
py .claude/gemini/handoff.py run .claude/gemini/tasks/<id>.json --dry-run   # check the prompt first
py .claude/gemini/handoff.py run .claude/gemini/tasks/<id>.json
```
The working tree must be clean, so commit or stash first. The wrapper writes `.claude/gemini/results/<id>.md`, which records the outcome (VERIFIED, CHECK_FAILED, BLOCKED or FAILED), Gemini's reply, the diff and the verification output. It never commits.

## Reviewing
At session start, a SessionStart hook lists results still awaiting review. For each one:
1. Read the diff against the spec and the packet rules.
2. Keep the change or `git checkout -- <file>`.
3. Run `py .claude/gemini/handoff.py reviewed <id>`.

Commits stay `Packet <id>: <title>`, made by Claude after review.
