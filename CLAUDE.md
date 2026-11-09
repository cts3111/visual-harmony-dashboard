# Project Context

Streamlit dashboard + live demo for the harmony-analyzer research. Self-contained project (own copies of pipeline code, models, and data), destined for the user's personal site.

# Rules

- Always ask clarifying questions before starting a complex task
- Show your plan and steps before executing
- Keep reports and summaries concise — bullet points over paragraphs
- Keep the story file `learning-story.md` updated as you build — log what you did, why, and key decisions as they happen
- README.md is the employer-facing summary; learning-story.md holds the decision/action log
- This project has a dedicated subagent: `dashboard-agent` (see `../.claude/agents/dashboard-agent.md`)

# Setup

- Run locally: `../harmony-analyzer/.venv/bin/python -m streamlit run app.py`
- Cloud deploy pins Python 3.11 (`runtime.txt`); dependencies in `requirements.txt`
- Deploy steps and the website embed plan are in README.md
