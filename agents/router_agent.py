# agents/router_agent.py
import json
import phoenix as px
from openai import OpenAI
from config import load_openai_api_key
from opentelemetry.trace import Status, StatusCode
from tracer import tracer
from utils.tool_schema import tools, tool_implementations
from opentelemetry.trace import get_current_span
#from experiment_config import CURRENT_VERSION, EXPERIMENT_NAME, get_current_version_info

INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
SPAN_KIND = "span_kind"


client = OpenAI(api_key=load_openai_api_key())
MODEL = "gpt-4o"


def handle_tool_calls(tool_calls, messages):
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        if tool_name in tool_implementations:
            result = tool_implementations[tool_name](**function_args)
        else:
            result = f"Error: Tool '{tool_name}' is not implemented."

        messages.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        })
    return messages


SYSTEM_PROMPT = """
You are a research assistant coordinator responsible for routing user requests to specialized agents.

You have access to the following tools:
- insight_agent(topic: string): generates multiple structured research insights on a given topic.
- summary_agent(insights: string, topic: string): generates a concise summary based on topic and insights.

IMPORTANT: You must respond only by calling one of the tools above using OpenAI function calling format.
Do NOT answer directly in natural language.

If the user input is a topic keyword, call insight_agent with that topic.
If the user input is a set of insights, call summary_agent with insights and topic.

Your response must be a JSON function call with the tool name and parameters.

Example function call:
{
  "name": "insight_agent",
  "arguments": {
    "topic": "Edge AI"
  }
}
"""
@tracer.chain()
def run_agent(user_input):
    span = get_current_span()

    # Add experiment metadata to span
    #version_info = get_current_version_info()
    #span.set_attribute("experiment.version", CURRENT_VERSION)
    #span.set_attribute("experiment.name", EXPERIMENT_NAME)
    #span.set_attribute("experiment.description", version_info.get("description", ""))
    #span.set_attribute("experiment.features", str(version_info.get("features", [])))

    print("Running router agent with input:", user_input)
    if not isinstance(user_input, str):
        raise ValueError("Input must be a string.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    print("Making router call to OpenAI")

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message
    messages.append(message)

    tool_calls = message.tool_calls
    print("Received response with tool calls:", bool(tool_calls))


    if tool_calls:
        tool_data =[
            {
                "name":tc.function.name,
                "arguments":tc.function.arguments
            } for tc in tool_calls
        ]

        span.set_attribute("llm.tools", json.dumps(tool_data))
        span.set_attribute(OUTPUT_VALUE, str(tool_calls))
        span.set_status(StatusCode.OK)

        # Return tool data WITHOUT executing - let caller execute the tool
        return {
            "tool_data": tool_data,
            "tool_calls": tool_calls,
            "messages": messages
        }

    else:
        span.set_attribute(OUTPUT_VALUE, message.content)
        span.set_status(StatusCode.OK)
        return {
            "tool_data": None,
            "result": message.content
        }
            
@tracer.chain()
def execute_tool_call(router_result):
    """Execute the tool call returned by router"""
    if not router_result.get("tool_data"):
        return router_result.get("result")

    tool_calls = router_result.get("tool_calls")
    messages = router_result.get("messages")

    if tool_calls and messages:
        messages = handle_tool_calls(tool_calls, messages)
        tool_result = messages[-1]["content"]
        tool_name = router_result["tool_data"][0]["name"]
        print(f"Tool executed: {tool_name}, result length: {len(tool_result)}")
        return tool_result

    return None

@tracer.chain()
def start_main_span(user_input):
    print("Starting main span with user_input:", user_input)
    ret = run_agent(user_input)
    return ret

# Test
if __name__ == "__main__":
    topic = "Edge AI"
    result = start_main_span(topic)
    print("Router agent result:", result)

    span = get_current_span()
    if span.is_recording():
        print("Attributes recorded in span:")
        print(span.attributes)