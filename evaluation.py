# Evaluate Routers
import warnings
import os
warnings.filterwarnings("ignore")
import phoenix as px
import pandas as pd
import json
from tqdm import tqdm
from phoenix.evals import(
    TOOL_CALLING_PROMPT_TEMPLATE,
    PromptTemplate,
    llm_classify,
    OpenAIModel,
    AnthropicModel
)

from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery
from openinference.instrumentation import suppress_tracing
from dotenv import load_dotenv
load_dotenv()
from anthropic import Anthropic
import nest_asyncio
nest_asyncio.apply()

from config import load_openai_api_key
import streamlit as st
from utils.tool_schema import tools
from tracer import tracer
from typing import Optional
from datetime import datetime

PROJECT_NAME = "tracing-research-agent"

#---------------------------
# Helper: Extract Topic from Spans
#---------------------------

def extract_topic_from_trace(client: px.Client, trace_id: str) -> Optional[str]:
    try:
        # Query run_agent spans in this trace to get the original user input
        query = SpanQuery().where(
            f"name == 'run_agent' and context.trace_id == '{trace_id}'"
        ).select(
            user_input="input.value"
        )

        df = client.query_spans(query, project_name=PROJECT_NAME)

        if not df.empty and 'user_input' in df.columns:
            # Get the first user input (which should be the topic)
            first_input = df.iloc[0]['user_input']

            if isinstance(first_input, str):
                if not first_input.startswith("Please summarize"):
                    return first_input.strip()

            # Try the second input if first one was a summary request
            if len(df) > 1:
                second_input = df.iloc[1]['user_input']
                if isinstance(second_input, str) and not second_input.startswith("Please summarize"):
                    return second_input.strip()

        return None

    except Exception as e:
        print(f"Warning: Could not extract topic from trace: {e}")
        return None

#---------------------------
# 1. Tool Calling Evaluation
#---------------------------

def tool_calling_eval():
    """
    Evaluate tool calling accuracy for all run_agent spans in database
    """
    client = px.Client()

    # Query all run_agent spans
    query = SpanQuery().where("name == 'run_agent'").select()
    df = client.query_spans(query, project_name=PROJECT_NAME)

    if df.empty:
        print("No run_agent spans found in database")
        return

    print(f"\nFound {len(df)} run_agent spans")

    # Extract and rename columns manually
    # Try different possible column names
    question_col = None
    if "input.value" in df.columns:
        question_col = "input.value"
    elif "attributes.input.value" in df.columns:
        question_col = "attributes.input.value"

    tool_call_col = None
    if "llm.tools" in df.columns:
        tool_call_col = "llm.tools"
    elif "attributes.llm.tools" in df.columns:
        tool_call_col = "attributes.llm.tools"

    span_id_col = None
    if "context.span_id" in df.columns:
        span_id_col = "context.span_id"
    elif "context_span_id" in df.columns:
        span_id_col = "context_span_id"

    # Debug: print available columns
    print(f"Available columns: {df.columns.tolist()[:10]}...")

    if not question_col:
        print("Warning: No 'input.value' column found. Setting to empty string.")
        df["question"] = ""
    else:
        df["question"] = df[question_col]

    if not tool_call_col:
        print("Error: No 'llm.tools' column found in spans.")
        print(f"All columns: {df.columns.tolist()}")
        return
    else:
        df["tool_call"] = df[tool_call_col]

    if not span_id_col:
        print("Warning: No 'context.span_id' column found.")
        # Try to use index as fallback
        df["span_id"] = df.index
    else:
        df["span_id"] = df[span_id_col]

    df["tool_call"] = df["tool_call"].fillna("No tool used")
    df = df[df["tool_call"] != "No tool used"]

    print(f"Spans with tool calls: {len(df)}")

    if df.empty:
        print("No tool call data found")
        return

    # Save span_id before llm_classify (it gets lost in the process)
    if 'span_id' in df.columns:
        span_ids = df['span_id'].copy()
    else:
        print(f"Warning: No 'span_id' in original df. Columns: {df.columns.tolist()}")
        span_ids = None

    # Evaluating Tool Calling
    with suppress_tracing():
        eval_df = llm_classify(
            dataframe = df[['question', 'tool_call']],  # Only pass needed columns
            template = TOOL_CALLING_PROMPT_TEMPLATE.template[0].template.replace(
                "{tool_definitions}", json.dumps(tools).replace("{", '"').replace("}", '"')),
            rails = ['correct', 'incorrect'],
            model=OpenAIModel(model="gpt-4o"),
        )

    # Join eval results with original df (which has span_id)
    result_df = df.join(eval_df)

    # Phoenix requires 'context.span_id' column for SpanEvaluations
    if 'context.span_id' not in result_df.columns:
        if 'span_id' in result_df.columns:
            result_df['context.span_id'] = result_df['span_id']
        elif span_ids is not None:
            # Restore from saved span_ids
            result_df['context.span_id'] = span_ids
        else:
            print(f"Available columns: {result_df.columns.tolist()}")
            raise ValueError("No 'context.span_id' column found")

    result_df['score'] = result_df["label"].apply(lambda x: 1 if x =='correct' else 0)



    # Save the evaluation results
    px.Client().log_evaluations(
        SpanEvaluations(
            eval_name="Tool Calling Eval",
            dataframe=result_df
        )
    )


    #print(tool_call_eval[["label", "explanation"]].head())

    accuracy = (result_df["label"] == "correct").mean()
    print(f"Tool calling accuracy: {accuracy:.2%}")


    # Show incorrect cases
    incorrect_cases = result_df[result_df["label"] == "incorrect"]

    required_cols = ["question", "tool_call", "explanation"]
    missing_cols = [col for col in required_cols if col not in incorrect_cases.columns]
    if missing_cols:
        print(f"Missing columns :{missing_cols}")
        print("Available columns:", incorrect_cases.columns.tolist())
    else:
        print(incorrect_cases[required_cols])

    # Save CSV with error handling
    csv_filename = "tool_call_eval_result.csv"
    try:
        result_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"Tool Calling evaluation saved to {csv_filename}\n")
    except PermissionError:
        # File is open, use timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"tool_call_eval_result_{timestamp}.csv"
        result_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"Original file is open. Saved to {csv_filename}\n")

#-------------------------
# 2. Insight Hallucination
#-------------------------

    # See all of columns
def insight_hallucination_eval(topic: Optional[str] = None):
    """
    Evaluate insight hallucination for all run_insight_agent spans in database

    Args:
        topic: Research topic (if None, will extract from trace)
    """
    client = px.Client()

    # Query all run_insight_agent spans
    query = SpanQuery().where("name == 'run_insight_agent'").select()
    df_insight = client.query_spans(query, project_name=PROJECT_NAME)

    if df_insight.empty:
        print("No run_insight_agent spans found in database")
        return

    print(f"\nFound {len(df_insight)} insight spans to evaluate")

    # Extract topic from trace if not provided
    if topic is None:
        trace_id = df_insight.iloc[0].get('context.trace_id')
        if trace_id:
            topic = extract_topic_from_trace(client, trace_id)
            if topic:
                print(f"Extracted topic from trace: '{topic}'")
            else:
                print("Warning: Could not extract topic from trace, using 'Unknown Topic'")
                topic = "Unknown Topic"
        else:
            print("Warning: No trace_id found, using 'Unknown Topic'")
            topic = "Unknown Topic"

    # Get insights contents
    insight_df = df_insight[df_insight["name"] == "run_insight_agent"][["context.span_id", "attributes.output.value"]]
    insight_df = insight_df.rename(columns={"attributes.output.value": "insights"})
    print(f"Insights to evaluate: {len(insight_df)}")



    eval_df = insight_df.reset_index(drop=True).copy()
    eval_df["topic"] = topic
    eval_df["insights"] = eval_df["insights"].str.replace("content='","", regex=False)
    eval_df["insights"] = eval_df["insights"].str.strip("'")

    # LLM-as-a-judge
    prompt = PromptTemplate("""
    You are a scientific research evaluator.
    You are given:
    - A research topic provided by the user.
    - AI-generated research insights in response to that topic.

    Evaluate whether the insights:
    1. Are relevant and focused on the given research topic.
    2. Contain any hallucinated content (fabricated or unsupported claims, papers, or citations).
    3. Are plausible and grounded in the topic context.

    ---

    Research Topic:
    {topic}

    AI-Generated Insights:
    {insights}

    ---

   Output strictly one word: "yes" or "no"

    """)

    with suppress_tracing():
        classified = llm_classify(
        dataframe=eval_df,
        template=prompt,
        rails=["yes", "no"],
        model=OpenAIModel(model="gpt-4o"),
      #  input_columns=["topic", "insights"]
    )

    result_df = pd.concat([eval_df, classified], axis=1)

    # Phoenix requires 'context.span_id' column for SpanEvaluations
    if 'context.span_id' not in result_df.columns:
        if 'context_span_id' in result_df.columns:
            result_df['context.span_id'] = result_df['context_span_id']
        else:
            raise ValueError("No 'context.span_id' column found")

    result_df["hallucinated"] = result_df["label"].apply(lambda x: 1 if x == "yes" else 0)

    #print(classified[["topic", "insights", "label", "hallucinated"]].head())


    px.Client().log_evaluations(
        SpanEvaluations(
            eval_name="Insights Hallucination Eval",
            dataframe=result_df
        )
    )

    # Save CSV with error handling
    csv_filename = "insights_hallucination_eval_result.csv"
    try:
        result_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"Insights Hallucination results saved to {csv_filename}!")
    except PermissionError:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"insights_hallucination_eval_result_{timestamp}.csv"
        result_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"Original file is open. Saved to {csv_filename}")

    print(f"Insights Hallucination rate: {result_df['hallucinated'].mean():.2%}")

    #-------------------------
    # 3. Summary Hallucination
    #-------------------------

def summary_hallucination_eval(topic: Optional[str] = None):
    """
    Evaluate summary hallucination for all run_summary_agent spans in database

    Args:
        topic: Research topic (if None, will extract from trace)
    """
    client = px.Client()

    # Query all run_summary_agent spans
    query = SpanQuery().where("name == 'run_summary_agent'").select()
    df_summary = client.query_spans(query, project_name=PROJECT_NAME)

    if df_summary.empty:
        print("No run_summary_agent spans found in database")
        return

    print(f"\nFound {len(df_summary)} summary spans to evaluate")

    # Extract topic from trace if not provided
    if topic is None:
        trace_id = df_summary.iloc[0].get('context.trace_id')
        if trace_id:
            topic = extract_topic_from_trace(client, trace_id)
            if topic:
                print(f"Extracted topic from trace: '{topic}'")
            else:
                print("Warning: Could not extract topic from trace, using 'Unknown Topic'")
                topic = "Unknown Topic"
        else:
            print("Warning: No trace_id found, using 'Unknown Topic'")
            topic = "Unknown Topic"

    # Get summary contents
    summary_df = df_summary[df_summary["name"] == "run_summary_agent"][["context.span_id", "attributes.output.value"]]
    summary_df = summary_df.rename(columns={"attributes.output.value": "summary"})

    print(f"Summaries to evaluate: {len(summary_df)}")

    eval_df = summary_df.reset_index(drop=True).copy()
    eval_df["topic"] = topic
    eval_df["summary"] = eval_df["summary"].str.replace("```markdown", "", regex=False)
    eval_df["summary"] = eval_df["summary"].str.strip("`").str.strip()
    eval_df["summary"] = eval_df["summary"].fillna("")


    # LLM-as-a-judge

    prompt = PromptTemplate("""
    You are a scientific research evaluator.
    You are given:
    - A research topic provided by the user.
    - AI-generated research summary in response to that topic.

    Evaluate whether the summary:
    1. Is relevant and focused on the given research topic.
    2. Contain any hallucinated content (fabricated or unsupported claims, papers, or citations).
    3. Is plausible and grounded in the topic context.

    ---

    Research Topic:
    {topic}

    AI-Generated Summary:
    {summary}

    ---

   Output strictly one word: "yes" or "no"
    """)


    with suppress_tracing():
        classified = llm_classify(
        dataframe=eval_df,
        template=prompt,
        rails=["yes", "no"],
        model=OpenAIModel(model="gpt-4o"),
    )

    result_df = pd.concat([eval_df, classified], axis=1)

    # Phoenix requires 'context.span_id' column for SpanEvaluations
    if 'context.span_id' not in result_df.columns:
        if 'context_span_id' in result_df.columns:
            result_df['context.span_id'] = result_df['context_span_id']
        else:
            raise ValueError("No 'context.span_id' column found")

    result_df["hallucinated"] = result_df["label"].apply(lambda x: 1 if x == "yes" else 0)

   # print(summary_eval[["topic", "summary", "label", "hallucinated"]].head())


    px.Client().log_evaluations(
        SpanEvaluations(
            eval_name="Summary Hallucination Eval",
            dataframe=result_df
        )
    )

    # Save CSV with error handling
    csv_filename = "summary_hallucination_eval_result.csv"
    try:
        result_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"Summary Hallucination results saved to {csv_filename}!")
    except PermissionError:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"summary_hallucination_eval_result_{timestamp}.csv"
        result_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"Original file is open. Saved to {csv_filename}")

    print(f"Summary Hallucination rate: {result_df['hallucinated'].mean():.2%}")

   # print("Logged evaluations:", client.spans.list_span_annotations())

def run_full_evaluation_pipeline(topic: str):
    """
    Run the complete evaluation pipeline for any research topic

    Args:
        topic: Research topic to evaluate (e.g., "Edge AI", "Quantum Computing", etc.)
    """
    from agents.router_agent import run_agent, execute_tool_call
    from tracer import tracer

    # Wrap the entire workflow in a traced function to keep everything in one trace
    @tracer.chain()
    def run_research_workflow(research_topic):
        """Run the complete research workflow in a single trace"""
        print(f"\n1. Calling router with topic: {research_topic}")
        result_1 = run_agent(research_topic)
        print(f"   Router decided to call: {result_1['tool_data'][0]['name']}")

        # Execute the tool
        insights = execute_tool_call(result_1)
        print(f"   Insights generated (length: {len(insights)})")

        print(f"\n2. Calling router with insights for summary")
        result_2 = run_agent(f"Please summarize these insights about {research_topic}: {insights[:500]}")
        print(f"   Router decided to call: {result_2['tool_data'][0]['name']}")

        # Execute the tool
        summary = execute_tool_call(result_2)
        print(f"   Summary generated (length: {len(summary)})")

        return insights, summary

    print("=" * 80)
    print(f"STEP 1: Running agents for topic: {topic}")
    print("=" * 80)

    insights, summary = run_research_workflow(topic)

    print("\nAgent execution completed!")

    print("\n" + "=" * 80)
    print("STEP 2: Running evaluations on ALL spans")
    print("=" * 80)

    print("\n1. Evaluating tool calling ...")
    tool_calling_eval()

    print("\n2. Evaluating insight hallucination ...")
    insight_hallucination_eval(topic)

    print("\n3. Evaluating summary hallucination ...")
    summary_hallucination_eval(topic)

    print("\n" + "=" * 80)
    print(f"ALL EVALUATIONS COMPLETED FOR TOPIC: {topic}")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - tool_call_eval_result.csv")
    print("  - insights_hallucination_eval_result.csv")
    print("  - summary_hallucination_eval_result.csv")
    print("\nCheck Phoenix UI at: http://localhost:6006")

    return {
        "topic": topic,
        "insights": insights,
        "summary": summary
    }


if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("Agent Evaluation Tool")
    print("=" * 80)

    if len(sys.argv) > 1:
        # Run full pipeline with provided topic
        topic = " ".join(sys.argv[1:])
        print(f"\nRunning full pipeline for topic: {topic}")
        print("=" * 80 + "\n")
        run_full_evaluation_pipeline(topic)
    else:

        print("\nEvaluating ALL spans from Phoenix")
        print("(Topics will be auto-extracted from traces)")
        print("=" * 80 + "\n")

        print("1. Evaluating tool calling (all traces)...")
        tool_calling_eval()

        print("\n2. Evaluating insight hallucination (all traces)...")
        insight_hallucination_eval()

        print("\n3. Evaluating summary hallucination (all traces)...")
        summary_hallucination_eval()

        print("\n" + "=" * 80)
        print("ALL EVALUATIONS COMPLETED!")
        print("=" * 80)
        print("\nResults saved to:")
        print("  - tool_call_eval_result.csv")
        print("  - insights_hallucination_eval_result.csv")
        print("  - summary_hallucination_eval_result.csv")
        print("\nCheck Phoenix UI at: http://localhost:6006")

