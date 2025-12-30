"""code_audit_agent.py

Fast Code Auditor orchestrator (Python + LangChain sub-agents).

Key performance improvements vs. the earlier version:
- Removes the *extra* orchestrator LLM call (previously 1 orchestrator call + 3 checker calls).
- Runs the 3 checker agents in parallel (threaded) so the overall time is close to the slowest checker call.

Supports:
1) CLI:  python -m agents.code_audit_agent --path <file> --out report.html
2) Jupyter widget: from agents.code_audit_agent import audit_widget; audit_widget()

Output:
- HTML report with table: line ref, severity, issue, snippet, resolution
- Overall score (%)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import nbformat
from dotenv import load_dotenv

from agents.shared import Issue, issues_to_html
from agents.sub_agents.quality_checker_agent import QualityCheckerAgent
from agents.sub_agents.readability_checker_agent import ReadabilityCheckerAgent
from agents.sub_agents.performance_checker_agent import PerformanceCheckerAgent


# -----------------------------
# Types
# -----------------------------
ProgressCallback = Callable[[int, int, str], None]
# callback signature: (completed_steps, total_steps, message)

from dotenv import load_dotenv
load_dotenv()
   
# -----------------------------
# Code extraction helpers
# -----------------------------

def _extract_code_from_py(path: Path) -> Tuple[str, Dict[int, str]]:
    """Return code + line_ref mapping for .py files.

    mapping: combined_line_number -> line_ref string (same as number)
    """
    text = path.read_text(encoding="utf-8")
    mapping = {i: str(i) for i in range(1, len(text.splitlines()) + 1)}
    return text, mapping


def _extract_code_from_ipynb(path: Path) -> Tuple[str, Dict[int, str]]:
    """Extract code cells from a notebook and build a stable line reference mapping.

    We concatenate code cells with cell markers so reviewers can locate issues.
    line_ref is reported as: 'cell <n>:<line>' where n is 1-based code-cell index.
    """
    nb = nbformat.read(path, as_version=4)

    combined_lines: List[str] = []
    mapping: Dict[int, str] = {}

    code_cell_idx = 0
    combined_line_no = 0

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue

        code_cell_idx += 1
        header = f"# --- cell {code_cell_idx} ---"
        combined_lines.append(header)
        combined_line_no += 1
        mapping[combined_line_no] = f"cell {code_cell_idx}:0"

        cell_lines = (cell.get("source") or "").splitlines()
        for j, ln in enumerate(cell_lines, start=1):
            combined_lines.append(ln)
            combined_line_no += 1
            mapping[combined_line_no] = f"cell {code_cell_idx}:{j}"

        # blank line spacer
        combined_lines.append("")
        combined_line_no += 1
        mapping[combined_line_no] = f"cell {code_cell_idx}:{len(cell_lines)+1}"

    return "\n".join(combined_lines).strip("\n"), mapping


def extract_code(path: Path) -> Tuple[str, Dict[int, str]]:
    if path.suffix.lower() == ".py":
        return _extract_code_from_py(path)
    if path.suffix.lower() == ".ipynb":
        return _extract_code_from_ipynb(path)
    raise ValueError(f"Unsupported file type: {path.suffix}. Use .py or .ipynb")


def _remap_line_refs(issues: List[Issue], mapping: Dict[int, str]) -> List[Issue]:
    """Remap numeric line refs to notebook cell refs when possible."""
    out: List[Issue] = []
    for it in issues:
        if not it.line_ref.isdigit():
            out.append(it)
            continue
        n = int(it.line_ref)
        out.append(
            Issue(
                line_ref=mapping.get(n, it.line_ref),
                issue=it.issue,
                snippet=it.snippet,
                resolution=it.resolution,
                severity=it.severity,
            )
        )
    return out


def _dedupe_issues(issues: List[Issue]) -> List[Issue]:
    seen = set()
    out: List[Issue] = []
    for it in issues:
        k = it.key()
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def _score(issues: List[Issue]) -> int:
    """Compute a simple score out of 100.

    Weighted penalty:
    - minor: 2
    - major: 6
    - critical: 12
    Cap at 100.
    """
    weight = {"minor": 2, "major": 6, "critical": 12}
    penalty = sum(weight.get(i.severity, 4) for i in issues)
    return int(max(0, 100 - min(100, penalty)))


# -----------------------------
# Agents
# -----------------------------
_quality_agent = QualityCheckerAgent()
_readability_agent = ReadabilityCheckerAgent()
_performance_agent = PerformanceCheckerAgent()


# -----------------------------
# Core audit runner (fast)
# -----------------------------
def run_audit(
    path: str,
    out_html: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> str:
    """Run the audit and return HTML string.

    Performance:
    - No orchestrator LLM call.
    - Checker agents run in parallel (3 threads).
    """
    load_dotenv()

    total_steps = 5
    completed = 0

    def _tick(msg: str) -> None:
        nonlocal completed
        completed += 1
        if progress_cb:
            progress_cb(completed, total_steps, msg)

    p = Path(path).expanduser().resolve()
    code, mapping = extract_code(p)
    _tick("Loaded code")

    # Run 3 agents concurrently (each may call the LLM once)
    _tick("Running checkers (quality/readability/performance)")

    futures = {}
    collected: List[Issue] = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures[ex.submit(_quality_agent.analyze, code)] = "Quality"
        futures[ex.submit(_readability_agent.analyze, code)] = "Readability"
        futures[ex.submit(_performance_agent.analyze, code)] = "Performance"

        done = 0
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                res = fut.result()
                collected.extend(res)
            except Exception as e:
                collected.append(
                    Issue(
                        line_ref="N/A",
                        issue=f"{label} checker failed: {e}",
                        snippet="(tool error)",
                        resolution="Check LLM credentials / model access and retry.",
                        severity="major",
                    )
                )
            done += 1
            if progress_cb:
                # we don't advance 'steps' here; just stream status
                progress_cb(completed, total_steps, f"{label} checker completed ({done}/3)")

    _tick("Merging results")

    parsed = _remap_line_refs(_dedupe_issues(collected), mapping)
    score = _score(parsed)
    html = issues_to_html(parsed, score=score, source_path=str(p))
    _tick("Rendered HTML report")

    if out_html:
        Path(out_html).write_text(html, encoding="utf-8")
        _tick(f"Wrote report: {out_html}")
    else:
        # still tick final step for consistent progress
        _tick("Done")

    return html


# -----------------------------
# Jupyter widget helper
# -----------------------------
def audit_widget(default_path: str = "sample_test.ipynb") -> None:
    """Render a small UI in Jupyter for running audits with progress."""
    try:
        import ipywidgets as widgets
        from IPython.display import display, HTML
    except Exception as e:
        raise RuntimeError("ipywidgets/IPython required in notebooks. Install requirements.txt") from e

    path_in = widgets.Text(value=default_path, description="Path:", layout=widgets.Layout(width="70%"))
    out_name = widgets.Text(value="(inline)", description="Out:", layout=widgets.Layout(width="30%"))

    run_btn = widgets.Button(description="Run Code Audit", button_style="primary")
    status = widgets.HTML(value="<b>Status:</b> Ready")
    prog = widgets.IntProgress(value=0, min=0, max=5, description="Progress:", bar_style="info")
    prog.layout.width = "60%"

    out = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="10px", width="100%"))

    def _progress_cb(done: int, total: int, msg: str) -> None:
        prog.max = total
        prog.value = min(done, total)
        status.value = f"<b>Status:</b> {msg}"

    def _on_click(_):
        out.clear_output()
        prog.value = 0
        status.value = "<b>Status:</b> Starting..."
        with out:
            try:
                # If user enters a real filename in Out, write file; otherwise inline.
                out_path = None if out_name.value.strip() in ("", "(inline)", "inline") else out_name.value.strip()
                html = run_audit(path_in.value, out_html=out_path, progress_cb=_progress_cb)
                display(HTML(html))
                status.value = "<b>Status:</b> Completed"
                prog.bar_style = "success"
            except Exception as ex:
                status.value = f"<b>Status:</b> Error: {ex}"
                prog.bar_style = "danger"
                print(f"Error: {ex}")

    run_btn.on_click(_on_click)
    display(widgets.VBox([widgets.HBox([path_in, out_name, run_btn]), widgets.HBox([prog, status]), out]))


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fast code auditor for .py and .ipynb")
    parser.add_argument("--path", required=True, help="Path to .py or .ipynb file")
    parser.add_argument("--out", default="code_audit_report.html", help="Output HTML file path")
    args = parser.parse_args()

    # Rich progress bar for terminal
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task("Auditing", total=5)

        def _progress_cb(done: int, total: int, msg: str) -> None:
            # Keep the task total in sync (defensive)
            progress.update(task_id, total=total)
            # Set description to current stage
            progress.update(task_id, description=f"Auditing: {msg}")
            progress.update(task_id, completed=min(done, total))

        run_audit(args.path, out_html=args.out, progress_cb=_progress_cb)

    print(f"Report written to: {args.out}")


if __name__ == "__main__":
    main()
