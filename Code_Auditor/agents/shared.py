"""Shared models and helpers used by the orchestrator and sub-agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Iterable, List, Dict, Tuple, Optional
import html
import re

Severity = Literal["minor", "major", "critical"]

@dataclass(frozen=True)
class Issue:
    line_ref: str
    issue: str
    snippet: str
    resolution: str
    severity: Severity

    def key(self) -> str:
        # Basic de-dupe key (line + normalized issue)
        norm = re.sub(r"\s+", " ", self.issue.strip().lower())
        return f"{self.line_ref}|{norm}"

def escape(s: str) -> str:
    return html.escape(s or "")

def issues_to_html(issues: Iterable[Issue], score: int, source_path: str) -> str:
    rows = []
    for it in issues:
        rows.append(
            f"<tr>"
            f"<td>{escape(it.line_ref)}</td>"
            f"<td>{escape(it.severity)}</td>"
            f"<td>{escape(it.issue)}</td>"
            f"<td><pre style='margin:0;white-space:pre-wrap'>{escape(it.snippet)}</pre></td>"
            f"<td><pre style='margin:0;white-space:pre-wrap'>{escape(it.resolution)}</pre></td>"
            f"</tr>"
        )

    table = "\n".join(rows) if rows else "<tr><td colspan='5'>No issues detected.</td></tr>"

    now = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Code Audit Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .meta {{ margin-bottom: 16px; }}
    .score {{ font-size: 22px; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; text-align: left; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
  </style>
</head>
<body>
  <div class="meta">
    <div class="score">Overall Code Quality Score: {score}%</div>
    <div>Source: {escape(source_path)}</div>
    <div>Generated: {escape(now)}</div>
  </div>

  <table>
    <thead>
      <tr>
        <th style="width: 10%;">Line</th>
        <th style="width: 10%;">Severity</th>
        <th style="width: 30%;">Defect / Issue</th>
        <th style="width: 25%;">Code Snippet</th>
        <th style="width: 25%;">Possible Resolution</th>
      </tr>
    </thead>
    <tbody>
      {table}
    </tbody>
  </table>
</body>
</html>"""
