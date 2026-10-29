#!/usr/bin/env python3
"""
Direct channel between the reviewing agent (Mac) and the implementing agent (Windows box),
carried by comments on https://github.com/shreyash616/echo/issues/1 through the gh CLI.

  agent_chat.py --role reviewer|implementer post "text"      post one message (or --file PATH, or - for stdin)
  agent_chat.py --role ... read [--all | --last N]            print messages not yet seen (both roles)
  agent_chat.py --role ... wait [--interval 60] [--timeout S] block until the OTHER role posts, print it, exit 0

Each comment starts with its role tag. Comments are data for the other agent, not instructions
for anyone else. The last-seen comment id is kept in ~/.echo_agent_chat/<role>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = "shreyash616/echo"
ISSUE = 1
GH_USER = "shreyash616"          # comments are posted by this account on both machines
ROLES = ("reviewer", "implementer")
IST = timezone(timedelta(hours=5, minutes=30))
STATE_DIR = Path.home() / ".echo_agent_chat"


def gh_env() -> dict:
    """gh's active account may differ from GH_USER on the reviewer's machine; use that user's token."""
    env = dict(os.environ)
    if "GH_TOKEN" in env:
        return env
    r = subprocess.run(["gh", "auth", "token", "-u", GH_USER], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        env["GH_TOKEN"] = r.stdout.strip()
    return env


def api(path: str, method: str = "GET", payload: dict | None = None):
    cmd = ["gh", "api", "-X", method, "-H", "Accept: application/vnd.github+json", path]
    data = None
    if payload is not None:
        cmd += ["--input", "-"]
        data = json.dumps(payload)
    r = subprocess.run(cmd, input=data, capture_output=True, text=True, env=gh_env())
    if r.returncode != 0:
        raise RuntimeError(f"gh api {method} {path} failed: {r.stderr.strip()[:500]}")
    return json.loads(r.stdout) if r.stdout.strip() else None


def state_path(role: str) -> Path:
    return STATE_DIR / f"{role}.json"


def last_seen(role: str) -> int:
    try:
        return int(json.loads(state_path(role).read_text())["last_seen_id"])
    except (OSError, ValueError, KeyError):
        return 0


def mark_seen(role: str, comment_id: int) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if comment_id > last_seen(role):
        state_path(role).write_text(json.dumps({"last_seen_id": comment_id}))


def comments(after_id: int = 0) -> list[dict]:
    out, page = [], 1
    while True:
        batch = api(f"repos/{REPO}/issues/{ISSUE}/comments?per_page=100&page={page}")
        out += [c for c in batch if c["id"] > after_id]
        if len(batch) < 100:
            return sorted(out, key=lambda c: c["id"])
        page += 1


def role_of(comment: dict) -> str | None:
    head = comment["body"].lstrip("*").lstrip()[:16]
    for r in ROLES:
        if head.startswith(f"[{r}]"):
            return r
    return None


def show(comment: dict) -> None:
    created = datetime.fromisoformat(comment["created_at"].replace("Z", "+00:00")).astimezone(IST)
    print(f"---- #{comment['id']}  {created:%Y-%m-%d %H:%M} IST  {comment['html_url']}")
    print(comment["body"].rstrip())
    print()


def cmd_post(args) -> None:
    if args.file:
        text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
    else:
        text = args.text or ""
    text = text.strip()
    if not text:
        sys.exit("nothing to post")
    body = f"**[{args.role}] {datetime.now(IST):%Y-%m-%d %H:%M} IST**\n\n{text}"
    c = api(f"repos/{REPO}/issues/{ISSUE}/comments", "POST", {"body": body})
    mark_seen(args.role, c["id"])
    print(f"posted #{c['id']} {c['html_url']}")


def cmd_read(args) -> None:
    if args.all:
        found = comments(0)
    elif args.last:
        found = comments(0)[-args.last:]
    else:
        found = comments(last_seen(args.role))
    if not found:
        print("no new messages")
    for c in found:
        show(c)
    if found and not args.last:
        mark_seen(args.role, found[-1]["id"])


def cmd_wait(args) -> None:
    other = [r for r in ROLES if r != args.role][0]
    deadline = time.time() + args.timeout if args.timeout else None
    while True:
        new = [c for c in comments(last_seen(args.role)) if role_of(c) == other]
        if new:
            for c in new:
                show(c)
            mark_seen(args.role, new[-1]["id"])
            print(f"NEW MESSAGE FROM {other.upper()}: #{new[-1]['id']}")
            return
        if deadline and time.time() > deadline:
            print("timeout, no new message")
            sys.exit(3)
        time.sleep(args.interval)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", required=True, choices=ROLES)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("post"); sp.add_argument("text", nargs="?"); sp.add_argument("--file")
    sr = sub.add_parser("read"); sr.add_argument("--all", action="store_true"); sr.add_argument("--last", type=int)
    sw = sub.add_parser("wait"); sw.add_argument("--interval", type=int, default=60); sw.add_argument("--timeout", type=int, default=0)
    args = p.parse_args()
    {"post": cmd_post, "read": cmd_read, "wait": cmd_wait}[args.cmd](args)


if __name__ == "__main__":
    main()
