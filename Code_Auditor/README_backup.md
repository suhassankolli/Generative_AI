# Code Auditor (Agentic Code Review)

This project is a **Python + LangChain agentic application** that performs automated code review for data-science code written in:
- **Python files** (`.py`)
- **Jupyter notebooks** (`.ipynb`)

It uses an LLM-based **orchestrator agent** (`agents/code_audit_agent.py`) that can decide which specialized tools to call:
1. **Code Quality Checker** (correctness, bad practices, unsafe patterns)
2. **Readability Checker** (maintainability, naming, complexity, structure)
3. **Performance Checker** (inefficiencies, vectorization opportunities, algorithmic pitfalls)

The final output is an **HTML report** containing a table of issues with:
- line number reference (or `cell X:Y` for notebooks)
- severity
- defect/issue
- offending code snippet
- recommended resolution

The report also includes an **overall code-quality score (%)**.

---

## Folder structure

```
Code_Auditor/
  agents/
    code_audit_agent.py
    prompt.py
    shared.py
    sub_agents/
      performance_checker_agent.py
      quality_checker_agent.py
      readability_checker_agent.py
  sample_test.ipynb
  README.md
  requirements.txt
```

---

## Prerequisites

- Python **3.10+** recommended
- An LLM API key (default implementation uses OpenAI via `langchain-openai`)
  - Set: `OPENAI_API_KEY` in your environment

---

## Installation

```bash
cd Code_Auditor
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage: Terminal / CLI

Run against a notebook or python file and write an HTML report:

```bash
python -m agents.code_audit_agent --path sample_test.ipynb --out report.html
```

Run against a Python file:

```bash
python -m agents.code_audit_agent --path your_script.py --out report.html
```

Optional: choose a model name:

```bash
python -m agents.code_audit_agent --path sample_test.ipynb --out report.html --model gpt-4o-mini
```

---

## Usage: Inside a Jupyter Notebook (widget)

In a notebook, after installing requirements, run:

```python
from agents.code_audit_agent import audit_widget
audit_widget("sample_test.ipynb")
```

A small widget UI will appear:
- **Path**: file to audit (`.py` or `.ipynb`)
- **Model**: model name
- **Run Code Audit**: generates and renders the HTML report inline

---

## How the agentic workflow works

### 1) Code extraction
`code_audit_agent.py` extracts code:
- `.py`: uses the file as-is with normal line numbers
- `.ipynb`: concatenates code cells and maintains a mapping so issues can be reported as **cell line references** (`cell 2:17`)

### 2) Orchestration (LLM agent)
The orchestrator is built with LangChain's **tool-calling agent**. The LLM chooses which tools to call depending on the code and the goal.

### 3) Specialized tools / sub-agents
Each tool delegates to a corresponding sub-agent:
- `QualityCheckerAgent` includes quick heuristics (e.g., `eval`, broad `except Exception`, mutable defaults) and an LLM review.
- `ReadabilityCheckerAgent` flags long lines, long functions, magic numbers, plus an LLM review.
- `PerformanceCheckerAgent` flags nested loops, `iterrows()`, and other common inefficiencies, plus an LLM review.

All three return issues in a common schema, which the orchestrator:
- merges
- de-duplicates
- computes a score
- renders to HTML

---

## Extending the solution

Common extensions:
- Add security checks (e.g., dependency vulnerabilities, secrets scanning)
- Add style/lint integrations (ruff, black, mypy) as additional tools
- Add a “fix-it” mode that proposes patches and creates PRs
- Add GitHub Actions CI step to auto-audit notebooks on pull requests

---

## Troubleshooting

- **LLM calls fail**: confirm `OPENAI_API_KEY` is set and the model is available to your account.
- **Notebook parsing fails**: ensure the `.ipynb` is valid JSON (try opening/saving it in Jupyter to normalize).
- **Widget not rendering**: install `ipywidgets` and enable in your Jupyter environment.

---

## License

Add your preferred license file (MIT / Apache-2.0) if needed.
