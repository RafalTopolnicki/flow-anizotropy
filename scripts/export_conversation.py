"""
Export the Claude Code session log for this project to a readable transcript.

Writes CONVERSATION.md (prose, tool calls collapsed, long output truncated) and
CONVERSATION.jsonl (the unabridged log). Picks the most recently modified
session file unless one is named explicitly.

Usage
-----
    python scripts/export_conversation.py
    python scripts/export_conversation.py --session <session-id>
    python scripts/export_conversation.py --max-output 4000
"""

import argparse
import datetime
import json
import re
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SESSION_DIR = Path.home() / ".claude" / "projects" / (
    "-" + str(PROJECT).lstrip("/").replace("/", "-"))


def fence(body: str, lang: str = "") -> str:
    """Fence longer than any backtick run inside, so output can't break out."""
    longest = max((len(r) for r in re.findall(r"`+", body)), default=2)
    bar = "`" * max(3, longest + 1)
    return f"{bar}{lang}\n{body.strip()}\n{bar}"


def blocks(msg):
    c = msg.get("content")
    if isinstance(c, str):
        return [{"type": "text", "text": c}]
    return c or []


def export(src: Path, out_md: Path, out_jsonl: Path, max_output: int) -> int:
    records = [json.loads(l) for l in src.open() if l.strip()]

    results = {}
    for d in records:
        if d.get("type") != "user":
            continue
        for b in blocks(d.get("message", {})):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                results[b.get("tool_use_id")] = c if isinstance(c, str) else str(c)

    out = [
        "# flow-anizotropy — session transcript\n",
        f"Session `{src.stem}`, exported {datetime.date.today()}.  ",
        "Full user prompts and assistant replies. Tool calls are collapsed into "
        "expandable sections showing the command or file path, with long output "
        f"truncated at {max_output} characters. The unabridged machine-readable "
        f"log is `{out_jsonl.name}`.\n",
        "`NOTES.md` is the distilled version — read that first; this file is the "
        "record of how the conclusions were reached.\n\n---\n",
    ]

    turn = 0
    for d in records:
        t = d.get("type")
        if t == "user":
            for b in blocks(d.get("message", {})):
                if not (isinstance(b, dict) and b.get("type") == "text"):
                    continue
                txt = b.get("text", "")
                if not txt.strip():
                    continue
                if txt.lstrip().startswith("<") and "system-reminder" in txt[:200]:
                    continue
                turn += 1
                out.append(f"\n## ▸ User ({turn})\n\n{txt.strip()}\n")
        elif t == "assistant":
            for b in blocks(d.get("message", {})):
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text", "").strip():
                    out.append(f"\n### Claude\n\n{b['text'].strip()}\n")
                elif b.get("type") == "tool_use":
                    name, inp = b.get("name"), b.get("input", {})
                    if name == "Bash":
                        head = f"🔧 Bash — {inp.get('description', '')}"
                        body = fence(inp.get("command", ""), "bash")
                    elif name in ("Write", "Edit", "Read"):
                        head = f"🔧 {name} — {inp.get('file_path', '')}"
                        body = ""
                    else:
                        head = f"🔧 {name}"
                        body = fence(json.dumps(inp)[:600], "json")
                    out.append(f"\n<details><summary>{head}</summary>\n\n{body}\n")
                    res = results.get(b.get("id"), "")
                    if res:
                        if len(res) > max_output:
                            res = (res[:max_output]
                                   + f"\n… [{len(res) - max_output} more characters truncated]")
                        out.append("\n" + fence(res) + "\n")
                    out.append("\n</details>\n")

    out_md.write_text("\n".join(out))
    shutil.copy(src, out_jsonl)
    return turn


def check(md: Path) -> bool:
    """
    Structural balance of the <details> blocks and the code fences.

    Counts only the exact forms `export` emits — a line starting with
    "<details><summary>" and a line that is exactly "</details>" — rather than
    searching for the tags anywhere. The transcript quotes those tags in prose,
    sometimes next to inline triple-backticks, which defeats both a naive
    substring count and any attempt to strip inline code spans first (backtick
    pairing goes wrong). Anchoring on the emitted form sidesteps all of it.
    """
    opens = closes = 0
    cur = None
    for line in md.read_text().split("\n"):
        m = re.match(r"^(`{3,})(.*)$", line)
        if m:
            run, rest = m.group(1), m.group(2).strip()
            if cur is None:
                cur = len(run)
                continue
            if len(run) >= cur and rest == "":
                cur = None
                continue
        if cur is not None:                 # inside a fenced block
            continue
        if line.startswith("<details><summary>"):
            opens += 1
        elif line.strip() == "</details>":
            closes += 1

    fences_ok = cur is None
    print(f"  fences closed: {fences_ok}    <details> {opens} / </details> {closes}")
    return fences_ok and opens == closes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", help="session id (default: most recent)")
    ap.add_argument("--max-output", type=int, default=1800)
    ap.add_argument("--outdir", default=str(PROJECT))
    args = ap.parse_args()

    if not SESSION_DIR.is_dir():
        raise SystemExit(f"no session directory at {SESSION_DIR}")

    if args.session:
        src = SESSION_DIR / f"{args.session}.jsonl"
        if not src.exists():
            raise SystemExit(f"no such session: {src}")
    else:
        logs = sorted(SESSION_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not logs:
            raise SystemExit(f"no .jsonl session logs in {SESSION_DIR}")
        src = logs[-1]

    outdir = Path(args.outdir)
    md = outdir / "CONVERSATION.md"
    jsonl = outdir / "CONVERSATION.jsonl"

    turns = export(src, md, jsonl, args.max_output)
    print(f"source  {src}")
    print(f"wrote   {md}  ({md.stat().st_size / 1024:.0f} KB, {turns} user turns)")
    print(f"wrote   {jsonl}  ({jsonl.stat().st_size / 1024**2:.1f} MB)")
    ok = check(md)
    print("markdown check", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
