"""code_audit_agent.py

LangChain LLM-based orchestrator that audits Python code from:
- a .py file
- a Jupyter notebook (.ipynb)

It exposes:
1) CLI entrypoint: python -m agents.code_audit_agent --path <file> --out report.html
2) Notebook widget: from agents.code_audit_agent import audit_widget; audit_widget()

The orchestrator is implemented as a tool-calling agent:
- quality_checker_tool
- readability_checker_tool
- performance_checker_tool

The LLM decides which tools to use. The app then merges tool outputs and generates an HTML report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import nbformat
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate,  MessagesPlaceholder

# ✅ LangChain 1.2.x: classic agents runtime lives in langchain-classic
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from agents.prompt import ORCHESTRATOR_INSTRUCTION
from agents.shared import Issue, issues_to_html
from agents.sub_agents.quality_checker_agent import QualityCheckerAgent
from agents.sub_agents.readability_checker_agent import ReadabilityCheckerAgent
from agents.sub_agents.performance_checker_agent import PerformanceCheckerAgent


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
        mapping[combined_line_no] = f"cell {code_cell_idx}:{len(cell_lines) + 1}"

    return "\n".join(combined_lines).strip("\n"), mapping


def extract_code(path: Path) -> Tuple[str, Dict[int, str]]:
    if path.suffix.lower() == ".py":
        return _extract_code_from_py(path)
    if path.suffix.lower() == ".ipynb":
        return _extract_code_from_ipynb(path)
    raise ValueError(f"Unsupported file type: {path.suffix}. Use .py or .ipynb")


def _remap_line_refs(issues: List[Issue], mapping: Dict[int, str]) -> List[Issue]:
    """Sub-agents typically report numeric line numbers from the combined code.
    This remaps to notebook 'cell X:Y' refs when possible.
    """
    out: List[Issue] = []
    for it in issues:
        # If already looks like 'cell X:Y' or non-numeric, keep.
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
    score = max(0, 100 - min(100, penalty))
    return int(score)


# -----------------------------
# Tool wrappers
# -----------------------------

_quality_agent = QualityCheckerAgent()
_readability_agent = ReadabilityCheckerAgent()
_performance_agent = PerformanceCheckerAgent()


@tool("code_quality_checker")
def code_quality_checker_tool(code: str) -> str:
    """Check code quality/correctness and return issues as JSON."""
    issues = _quality_agent.analyze(code)
    return json.dumps([asdict(i) for i in issues], ensure_ascii=False)


@tool("readability_checker")
def readability_checker_tool(code: str) -> str:
    """Check readability/maintainability and return issues as JSON."""
    issues = _readability_agent.analyze(code)
    return json.dumps([asdict(i) for i in issues], ensure_ascii=False)


@tool("performance_checker")
def performance_checker_tool(code: str) -> str:
    """Check performance and return issues as JSON."""
    issues = _performance_agent.analyze(code)
    return json.dumps([asdict(i) for i in issues], ensure_ascii=False)


TOOLS = [code_quality_checker_tool, readability_checker_tool, performance_checker_tool]


# -----------------------------
# Orchestrator
# -----------------------------

def run_audit(path: str, out_html: Optional[str] = None, model: str = "gpt-4o-mini") -> str:
    """Run the audit and return HTML string."""
    from dotenv import load_dotenv
    load_dotenv()
   

    p = Path(path).expanduser().resolve()
    code, mapping = extract_code(p)

    llm = ChatOpenAI(model=model, temperature=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ORCHESTRATOR_INSTRUCTION),
            (
                "human",
                "Audit this code from file '{source}'.\n\n"
                "You may call tools to find issues.\n"
                "Return only the final HTML report string.\n\n"
                "CODE:\n{code}",
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    

    # ✅ classic tool-calling agent + executor
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False)

    # Run agent (it will call tools and then respond)
    agent_html = executor.invoke({"code": code, "source": str(p)}).get("output", "")

    # Deterministic fallback: ensure we always return a valid HTML report
    if not agent_html.strip().lower().startswith("<!doctype html") and "<table" not in agent_html.lower():
        raw = []
        for t in (code_quality_checker_tool, readability_checker_tool, performance_checker_tool):
            raw.append(t.invoke(code))

        parsed: List[Issue] = []
        for blob in raw:
            try:
                data = json.loads(blob)
                for item in data:
                    parsed.append(Issue(**item))
            except Exception:
                continue

        parsed = _remap_line_refs(_dedupe_issues(parsed), mapping)
        score = _score(parsed)
        agent_html = issues_to_html(parsed, score=score, source_path=str(p))

    if out_html:
        Path(out_html).write_text(agent_html, encoding="utf-8")

    return agent_html


# -----------------------------
# Jupyter widget helper
# -----------------------------

def audit_widget(default_path: str = "sample_test.ipynb") -> None:
    """Render a small UI in Jupyter for running audits."""
    try:
        import ipywidgets as widgets
        from IPython.display import display, HTML
    except Exception as e:
        raise RuntimeError("ipywidgets/IPython required in notebooks. Install requirements.txt") from e

    path_in = widgets.Text(value=default_path, description="Path:", layout=widgets.Layout(width="80%"))
    model_in = widgets.Text(value="gpt-4o-mini", description="Model:", layout=widgets.Layout(width="50%"))

    run_btn = widgets.Button(description="Run Code Audit", button_style="primary")
    out = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="10px"))

    def _on_click(_):
        out.clear_output()
        with out:
            try:
                html = run_audit(path_in.value, out_html=None, model=model_in.value)
                display(HTML(html))
            except Exception as ex:
                print(f"Error: {ex}")

    run_btn.on_click(_on_click)
    display(widgets.VBox([widgets.HBox([path_in, model_in, run_btn]), out]))


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-based code auditor for .py and .ipynb")
    parser.add_argument("--path", required=True, help="Path to .py or .ipynb file")
    parser.add_argument("--out", default="code_audit_report.html", help="Output HTML file path")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name (provider-specific)")
    args = parser.parse_args()

    run_audit(args.path, out_html=args.out, model=args.model)
    print(f"Report written to: {args.out}")


if __name__ == "__main__":
    main()