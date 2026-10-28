"""
.claude/gemini/handoff.py: hand one tiny, fully specified task to the Gemini agent (agy CLI).

Claude is the primary agent. Gemini is used only when Claude's quota is nearly exhausted, and only
for micro-tasks whose outcome is fixed by the spec: an allow-list of files, exact steps, and a
verification command that this wrapper (not Gemini) runs. Gemini never commits; every result waits
for Claude's review.

  py .claude/gemini/handoff.py run <task.json> [--dry-run]
  py .claude/gemini/handoff.py pending          # SessionStart hook: list unreviewed results
  py .claude/gemini/handoff.py reviewed <id>    # mark a result reviewed

Task spec (JSON):
  {"id": "p2-sat-01", "goal": "...", "files": ["backend/app/audio/features.py"],
   "steps": ["...", "..."], "verify": ["py -m pytest -q tests/test_features_golden.py"],
   "model": "gemini-3.8-flash-high", "timeout_s": 1200}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / ".claude" / "gemini" / "results"
DEFAULT_MODEL = "gemini-3.8-flash-high"

RULES = """You are a helper on a gated project. Follow these rules exactly:
1. Edit ONLY these files: {files}. Do not create, rename or delete any other file.
2. Do not run shell commands, tests, git, installs or network calls. The caller verifies your work.
3. Do not commit. Do not refactor, reformat or touch code the steps do not name.
4. Do the steps in order, literally. If a step is ambiguous or impossible, change nothing and reply
   with a first line `BLOCKED: <reason>`.
5. When finished, reply with a first line `DONE` followed by one line per file changed."""


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


def changed_paths() -> set[str]:
    out = set()
    for line in git("status", "--porcelain", "--untracked-files=all").splitlines():
        p = line[3:].strip().strip('"')
        if " -> " in p:
            p = p.split(" -> ", 1)[1]
        if not p.startswith(".claude/gemini/results/"):
            out.add(p.replace("\\", "/"))
    return out


def build_prompt(task: dict) -> str:
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(task["steps"], 1))
    return (
        RULES.format(files=", ".join(task["files"]))
        + f"\n\nGOAL: {task['goal']}\n\nSTEPS:\n{steps}\n"
    )


def run(task_path: str, dry_run: bool) -> int:
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    for k in ("id", "goal", "files", "steps", "verify"):
        if not task.get(k):
            sys.exit(f"task spec missing '{k}'")
    allowed = {f.replace("\\", "/") for f in task["files"]}

    dirty = changed_paths()
    if dirty:
        sys.exit(f"working tree not clean, refusing so the diff is attributable: {sorted(dirty)}")

    agy = shutil.which("agy") or shutil.which("agy.exe")
    if not agy:
        sys.exit("agy CLI not found on PATH")
    prompt = build_prompt(task)
    cmd = [agy, "-p", prompt, "--model", task.get("model", DEFAULT_MODEL),
           "--mode", "accept-edits", "--sandbox", "--disable-slash-commands",
           "--print-timeout", f"{task.get('timeout_s', 1200)}s"]
    if dry_run:
        print(prompt)
        print("command:", " ".join(cmd[:1] + ["-p", "<prompt>"] + cmd[3:]))
        return 0

    head = git("rev-parse", "HEAD").strip()
    started = dt.datetime.now().isoformat(timespec="seconds")
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              timeout=task.get("timeout_s", 1200) + 60)
        reply, agy_rc = (proc.stdout + proc.stderr).strip(), proc.returncode
    except subprocess.TimeoutExpired:
        reply, agy_rc = "wrapper timeout", -1

    # Undo anything outside the allow-list; the tree was clean, so this only touches Gemini's work.
    stray = sorted(changed_paths() - allowed)
    for p in stray:
        if git("ls-files", p).strip():
            git("checkout", "--", p)
        else:
            (ROOT / p).unlink(missing_ok=True)

    diff = git("diff", "--", *sorted(allowed))
    verify_out = []
    for c in task["verify"]:
        v = subprocess.run(c, cwd=ROOT, shell=True, capture_output=True, text=True)
        verify_out.append((c, v.returncode, (v.stdout + v.stderr)[-6000:]))

    first = reply.splitlines()[0] if reply else ""
    if agy_rc != 0 or not first.startswith(("DONE", "BLOCKED")):
        status = "FAILED"
    elif first.startswith("BLOCKED"):
        status = "BLOCKED"
    elif stray or not diff.strip() or any(rc for _, rc, _ in verify_out):
        status = "CHECK_FAILED"
    else:
        status = "VERIFIED"

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{task['id']}.md"
    parts = [
        f"# Gemini task {task['id']}",
        f"- Review: PENDING_REVIEW",
        f"- Outcome: {status}",
        f"- Started: {started}; base commit {head}; agy exit {agy_rc}",
        f"- Allowed files: {', '.join(sorted(allowed))}",
        f"- Stray edits reverted: {', '.join(stray) or 'none'}",
        f"\n## Goal\n{task['goal']}",
        f"\n## Gemini reply\n```\n{reply[-6000:]}\n```",
        f"\n## Diff\n```diff\n{diff}\n```",
    ]
    for c, rc, o in verify_out:
        parts.append(f"\n## Verify: `{c}` (exit {rc})\n```\n{o}\n```")
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"{status}: {out}")
    return 0 if status == "VERIFIED" else 1


def pending() -> int:
    if not RESULTS.exists():
        return 0
    rows = []
    for f in sorted(RESULTS.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        if "- Review: PENDING_REVIEW" in text:
            outcome = next((l.split(": ", 1)[1] for l in text.splitlines() if l.startswith("- Outcome:")), "?")
            rows.append(f"  - {f.stem}: {outcome} ({f.relative_to(ROOT).as_posix()})")
    if rows:
        print("Gemini hand-off results awaiting Claude's review (uncommitted edits may be in the tree):")
        print("\n".join(rows))
        print("Review each diff against FIX_PLAN_2.md, then: py .claude/gemini/handoff.py reviewed <id>")
    return 0


def reviewed(task_id: str) -> int:
    f = RESULTS / f"{task_id}.md"
    f.write_text(f.read_text(encoding="utf-8").replace("- Review: PENDING_REVIEW", "- Review: REVIEWED", 1),
                 encoding="utf-8")
    print(f"marked reviewed: {f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("task")
    r.add_argument("--dry-run", action="store_true")
    sub.add_parser("pending")
    rv = sub.add_parser("reviewed")
    rv.add_argument("id")
    a = ap.parse_args()
    sys.exit(run(a.task, a.dry_run) if a.cmd == "run" else pending() if a.cmd == "pending" else reviewed(a.id))
