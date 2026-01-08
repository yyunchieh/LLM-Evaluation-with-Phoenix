# LLM-Evaluation-with-Phoenix
A research assistant agent system built with OpenAI GPT-4o and Phoenix Tracing, featuring a comprehensive evaluation framework.

## Project Structure

```
LLM-Evaluation-with-Phoenix/
├── agents/                  # Python package containing agent definitions 
│   ├── __init__.py
│   ├── router_agent.py      # Routes user intent to specialized agents
│   ├── insight_agent.py     # Generates research insights
│   └── summary_agent.py     # Summarizes research insights
├── utils/                   # Python package for utilities and tools
│   ├── __init__.py
│   ├── web_search.py        # Tavily web search integration
│   └── tool_schema.py       # OpenAI tool schemas
├── evaluation.py            # Evaluation framework (tool calling & hallucination)
├── tracer.py                # Phoenix tracing & OpenTelemetry setup
├── main.py                  # Streamlit UI
├── config.py                # Environment & configuration management
├── langgraph.json           # Declarative agent graph configuration & entry point
├── .env                     # Environment variables (create by yourself; contains API keys)
├── requirements.txt
└── README.md
```

## Features

### 1. Multi-Agent System
- **Router Agent**: Routes user requests to appropriate agents, logs tool usage for evaluation
- **Insight Agent**: Generates structured research insights including:
  - Domain introduction
  - List of research papers (Chicago Style citation format)
  - Future research directions
  - Research title suggestions
- **Summary Agent**: Summarizes multiple research insights

### 2. Web Search Integration
- Uses Tavily API for web search
- Reduces LLM hallucination (from 70-80%)
- Generates research content based on real search results

### 3. Comprehensive Evaluation Framework
- **Tool Calling Accuracy Evaluation**: Evaluates whether Router correctly selects tools
- **Insight Hallucination Evaluation**: Detects fabricated content in generated research insights
- **Summary Hallucination Evaluation**: Detects if summary deviates from original insights

### 4. Phoenix Tracing
- Complete tracing of all LLM calls
- Visualizes agent interaction flows
- Evaluation results automatically logged to Phoenix UI

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create your own `.env` file with the following content:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Usage

### Option 1: Streamlit UI

Launch the Streamlit interface:

```bash
streamlit run main.py
```

Features:
- Enter research topic
- Automatically generate research insights and summary
- Download results in Markdown format
- Save to `outputs/` folder (timestamped)

### Option 2: Evaluate Existing Spans

Evaluate all spans in the Phoenix database:

```bash
python evaluation.py
```

This executes three types of evaluation:
1. Tool calling accuracy evaluation
2. Insight hallucination evaluation
3. Summary hallucination evaluation

Results are saved to:
- `tool_call_eval_result.csv`
- `insights_hallucination_eval_result.csv`
- `summary_hallucination_eval_result.csv`

### Option 3: Run Full Workflow and Evaluate

Execute complete workflow for a specific topic and evaluate:

```bash
python evaluation.py "Edge AI"
```

This will:
1. Execute Router → Insight Agent → Summary Agent flow
2. Evaluate the newly generated spans
3. Save Markdown results and evaluation reports

## Phoenix UI

Phoenix UI automatically launches in the background, access at:

```
http://localhost:6006
```

In the UI you can see:
- Visualization of all traces
- LLM call details
- Evaluation result circles (showing accuracy and hallucination rates)
- Span attributes and timeline

## Evaluation Metrics

### 1. Tool Calling Accuracy
- Evaluates whether Router Agent correctly selects tools
- Uses GPT-4o as judge
- Output: correct / incorrect

### 2. Hallucination Rate
- Evaluates whether generated content contains fabricated or unsupported claims
- Checks authenticity of paper citations
- Output: yes (hallucinated) / no (not hallucinated)

## Key Files Explanation

### agents/insight_agent.py
- Uses ChatOpenAI (GPT-4o) to generate research insights
- Integrates Tavily Web Search to reduce hallucination
- Generates 5 sets of structured research insights
- Uses `@tracer.chain()` for execution tracing

### agents/router_agent.py
- Uses OpenAI Function Calling to route user requests
- Separates routing logic (`run_agent`) from execution logic (`execute_tool_call`)
- Records `llm.tools` attribute for evaluation

### agents/summary_agent.py
- Summarizes multiple research insights
- Extracts key points, research papers, trends, and future directions
- Purely based on provided insights, does not introduce new information

### Evaluation.py
Three main evaluation functions:

1. **`tool_calling_eval()`**
   - Queries all `run_agent` spans
   - Evaluates using `TOOL_CALLING_PROMPT_TEMPLATE`
   - Calculates accuracy rate

2. **`insight_hallucination_eval(topic)`**
   - Queries all `run_insight_agent` spans
   - Automatically extracts topic from trace
   - Evaluates whether content is fabricated

3. **`summary_hallucination_eval(topic)`**
   - Queries all `run_summary_agent` spans
   - Evaluates whether summary deviates from original insights

### utils/web_search.py
- Tavily API integration
- `web_search()`: General search function
- `web_search_for_research()`: Research topic-specific search

## Output Files

### Markdown Output
- Location: `outputs/YYYY-MM-DD_HH-MM-SS/Research_Insights_<topic>.md`
- Format: Markdown document containing research insights and summary

### Evaluation Result CSVs
- `tool_call_eval_result.csv`: Tool calling evaluation results
- `insights_hallucination_eval_result.csv`: Insight hallucination evaluation results
- `summary_hallucination_eval_result.csv`: Summary hallucination evaluation results

## Development Notes

1. **Span Tracing**: All major functions use the `@tracer.chain()` decorator
2. **Evaluation Logging**: Use `px.Client().log_evaluations()` to log evaluation results to Phoenix
3. **Attribute Setting**: Set `llm.tools` attribute on Router span for evaluation use
4. **Query Syntax**: Use Phoenix DSL (`SpanQuery()`) to query spans

## Tech Stack

- **LLM**: OpenAI GPT-4o
- **Framework**: OpenAI Python SDK, LangChain (ChatOpenAI wrapper)
- **Tracing**: Phoenix AI (OpenTelemetry)
- **Web Search**: Tavily API
- **UI**: Streamlit
- **Evaluation**: Phoenix Evals (LLM-as-a-judge)
