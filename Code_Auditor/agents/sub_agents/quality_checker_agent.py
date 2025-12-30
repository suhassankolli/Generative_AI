"""Quality checker sub-agent.

This agent focuses on correctness, defensive programming, and common Python pitfalls.
It uses:
- fast heuristic scans (AST + regex)
- optional LLM review to catch higher-level issues

It returns a list[Issue] (see agents.shared.Issue).
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


class QualityCheckerAgent:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        # For OpenAI, set env var: OPENAI_API_KEY
        from dotenv import load_dotenv
        load_dotenv()

        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # IMPORTANT: since we parse into _LLMIssues, the model should return an object:
        # { "issues": [ ... ] }
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_GOAL),
                (
                    "human",
                    "Perform a QUALITY/CORRECTNESS review. Identify bugs, risky patterns, missing error handling.\n"
                    "Return a JSON object with a single key 'issues' whose value is a list of issues.\n"
                    "Each issue must match this schema:\n"
                    "line_ref, issue, snippet, resolution, severity (minor|major|critical).\n\n"
                    "CODE:\n{code}",
                ),
            ]
        )

    def _heuristics(self, code: str) -> List[Issue]:
        issues: List[Issue] = []

        # Heuristic 1: broad exception catches
        for i, line in enumerate(code.splitlines(), start=1):
            if re.search(r"except\s+Exception\s*:", line):
                issues.append(
                    Issue(
                        line_ref=str(i),
                        issue="Catching bare Exception can hide real failures and makes debugging harder.",
                        snippet=line.strip(),
                        resolution="Catch specific exception types, and log/raise with context.",
                        severity="major",
                    )
                )

        # Heuristic 2: eval/exec usage
        for i, line in enumerate(code.splitlines(), start=1):
            if re.search(r"\beval\(|\bexec\(", line):
                issues.append(
                    Issue(
                        line_ref=str(i),
                        issue="Use of eval/exec is unsafe and can lead to code injection.",
                        snippet=line.strip(),
                        resolution="Avoid eval/exec. If parsing is needed, use ast.literal_eval or a safe parser.",
                        severity="critical",
                    )
                )

        # Heuristic 3: mutable default args
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for default in node.args.defaults:
                        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                            issues.append(
                                Issue(
                                    line_ref=str(node.lineno),
                                    issue="Mutable default argument can cause state to leak across calls.",
                                    snippet=f"def {node.name}(...):",
                                    resolution="Use None as default and initialize inside the function.",
                                    severity="major",
                                )
                            )
        except SyntaxError:
            # Let LLM handle malformed code; still report
            issues.append(
                Issue(
                    line_ref="N/A",
                    issue="Code contains syntax errors; static parsing failed.",
                    snippet="(unable to parse)",
                    resolution="Fix syntax errors first, then re-run audit.",
                    severity="critical",
                )
            )

        return issues

    def analyze(self, code: str, source_hint: Optional[str] = None) -> List[Issue]:
        issues = self._heuristics(code)

        # LLM review (structured output)
        chain = self.prompt | self.llm.with_structured_output(_LLMIssues)

        try:
            result: _LLMIssues = chain.invoke({"code": code})
            for it in result.issues:
                issues.append(Issue(**it.model_dump()))
        except Exception as e:
            issues.append(
                Issue(
                    line_ref="N/A",
                    issue=f"LLM quality review failed: {e}",
                    snippet="(tool error)",
                    resolution="Verify API key/model access and retry.",
                    severity="major",
                )
            )

        return issues