"""Readability checker sub-agent.

Focuses on maintainability: naming, structure, documentation, complexity, style.
Uses light heuristics + LLM structured extraction.

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


class ReadabilityCheckerAgent:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # IMPORTANT: since we parse into _LLMIssues, we ask the LLM to return:
        # { "issues": [ ... ] }
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_GOAL),
                (
                    "human",
                    "Perform a READABILITY/MAINTAINABILITY review. Identify confusing naming, long functions, "
                    "magic numbers, missing docs, unnecessary complexity.\n"
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

        # Heuristic 1: long lines
        for i, line in enumerate(lines, start=1):
            if len(line) > 120:
                issues.append(
                    Issue(
                        line_ref=str(i),
                        issue="Line exceeds 120 characters; hurts readability.",
                        snippet=line[:200],
                        resolution="Wrap the line, extract expressions into well-named variables, or format using black/ruff.",
                        severity="minor",
                    )
                )

        # Heuristic 2: long functions (rough maintainability signal)
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    end = getattr(node, "end_lineno", None) or node.lineno
                    length = end - node.lineno + 1
                    if length > 60:
                        issues.append(
                            Issue(
                                line_ref=str(node.lineno),
                                issue=f"Function '{node.name}' is long ({length} lines). Hard to test/maintain.",
                                snippet=f"def {node.name}(...):",
                                resolution="Refactor into smaller functions; isolate pure logic; add docstrings and unit tests.",
                                severity="major",
                            )
                        )
        except SyntaxError:
            # ignore here; quality agent will flag
            pass

        # Heuristic 3: magic numbers (very simple)
        for i, line in enumerate(lines, start=1):
            if re.search(r"[^\w](\d{3,}|0\.\d{3,})[^\w]", line) and "http" not in line:
                issues.append(
                    Issue(
                        line_ref=str(i),
                        issue="Possible magic number reduces readability/maintainability.",
                        snippet=line.strip(),
                        resolution="Replace with a named constant and explain its meaning.",
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
                    issue=f"LLM readability review failed: {e}",
                    snippet="(tool error)",
                    resolution="Verify API key/model access and retry.",
                    severity="major",
                )
            )

        return issues