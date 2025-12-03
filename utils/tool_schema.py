# utils/tool_schema.py

from agents.insight_agent import run_insight_agent
from agents.summary_agent import run_summary_agent

tools = [
    {
        "type": "function",
        "function": {
            "name": "insight_agent",
            "description": "Generate research insights about a given topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summary_agent",
            "description": "Summarize a set of insights about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "insights": {"type": "string"},
                    "topic": {"type": "string"}
                },
                "required": ["insights", "topic"]
            }
        }
    }
]

tool_implementations = {
    "insight_agent": run_insight_agent,
    "summary_agent": run_summary_agent
}
