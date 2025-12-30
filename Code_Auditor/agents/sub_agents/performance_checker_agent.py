"""Performance checker sub-agent.

Looks for common data-science performance pitfalls:
- Python loops over pandas/numpy data
- Recomputing expensive operations inside loops
- Not using vectorization / built-in methods
- Inefficient string concatenation in loops
- Unnecessary conversions/copies

Uses heuristics + LLM structured extraction.

Returns: list[Issue]
"""

from __future__ import annotations

from typing import List, Optional
import ast
import re

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.prompt import SYSTEM_GOAL
from agents.shared import Issue, Severity


class _LLMIssue(BaseModel):
    """Single issue returned by the LLM."""
    line_ref: str = Field(...)
    issue: str = Field(...)
    snippet: str = Field(...)
    resolution: str = Field(...)
    severity: Severity = Field(...)


class _LLMIssues(BaseModel):
    """Wrapper model so LangChain structured output has a concrete BaseModel target."""
    issues: List[_LLMIssue] = Field(default_factory=list)


class PerformanceCheckerAgent:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        # If you use dotenv in the project, keep behavior consistent with other sub-agents
        from dotenv import load_dotenv
        load_dotenv()

        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # IMPORTANT: we parse into _LLMIssues, so we ask the LLM to return:
        # { "issues": [ ... ] }
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_GOAL),
                (
                    "human",
                    "Perform a PERFORMANCE review. Identify inefficiencies and suggest optimizations "
                    "(vectorization, caching, algorithmic improvements).\n"
                    "Return a JSON object with a single key 'issues' whose value is a list of issues.\n"
                    "Each issue must match this schema:\n"
                    "line_ref, issue, snippet, resolution, severity (minor|major|critical).\n\n"
                    "CODE:\n{code}",
                ),
            ]
        )

    def _heuristics(self, code: str) -> List[Issue]:
        issues: List[Issue] = []
        lines = code.splitlines()

        # Heuristic 1: string concatenation in loops
        # Note: can't always detect "in loop" from a single line, but we can still flag common pattern.
        for i, line in enumerate(lines, start=1):
            if "+=" in line and ("f\"" in line or "format(" in line or "str(" in line):
                issues.append(
                    Issue(
                        line_ref=str(i),
                        issue="Potential slow string concatenation (+=) can become O(n^2) in loops.",
                        snippet=line.strip(),
                        resolution="Accumulate strings in a list and use ''.join(parts) once at the end.",
                        severity="minor",
                    )
                )

        # Heuristic 2: nested loops (rough but useful)
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    for inner in ast.walk(node):
                        if inner is not node and isinstance(inner, ast.For):
                            issues.append(
                                Issue(
                                    line_ref=str(node.lineno),
                                    issue="Nested for-loops may be slow for large datasets.",
                                    snippet="for ...:\n    for ...:",
                                    resolution="Consider vectorization (numpy/pandas), batching, or optimized libraries.",
                                    severity="major",
                                )
                            )
                            break
        except SyntaxError:
            # If parsing fails, LLM might still provide useful performance notes
            pass

        # Heuristic 3: pandas row iteration patterns
        for i, line in enumerate(lines, start=1):
            if "iterrows(" in line or "itertuples(" in line:
                issues.append(
                    Issue(
                        line_ref=str(i),
                        issue="Row-wise iteration over pandas DataFrame is slow for large data.",
                        snippet=line.strip(),
                        resolution="Prefer vectorized ops, merge/join, boolean masks, or numpy arrays. "
                                   "If iteration is unavoidable, prefer itertuples() over iterrows().",
                        severity="major",
                    )
                )

        # Heuristic 4: list building + sum(listcomp) pattern
        # Example: sum([x*x for x in data]) -> use generator: sum(x*x for x in data)
        for i, line in enumerate(lines, start=1):
            if re.search(r"sum\(\s*\[.+for.+in.+\]\s*\)", line):
                issues.append(
                    Issue(
                        line_ref=str(i),
                        issue="Using sum([...]) builds an intermediate list; can waste memory on large data.",
                        snippet=line.strip(),
                        resolution="Use a generator: sum(x*x for x in data) or numpy vectorization when applicable.",
                        severity="minor",
                    )
                )

        return issues

    def analyze(self, code: str, source_hint: Optional[str] = None) -> List[Issue]:
        issues = self._heuristics(code)

        # LLM review (structured output) — use wrapper model, not list[_LLMIssue]
        chain = self.prompt | self.llm.with_structured_output(_LLMIssues)

        try:
            result: _LLMIssues = chain.invoke({"code": code})
            for it in result.issues:
                issues.append(Issue(**it.model_dump()))
        except Exception as e:
            issues.append(
                Issue(
                    line_ref="N/A",
                    issue=f"LLM performance review failed: {e}",
                    snippet="(tool error)",
                    resolution="Verify API key/model access and retry.",
                    severity="major",
                )
            )

        return issues