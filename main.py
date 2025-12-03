# main.py

import os

from datetime import datetime
# environment setting
os.environ["PHOENIX_GRPC_PORT"] = "4317" 

import streamlit as st
from agents.insight_agent import run_insight_agent
from agents.summary_agent import run_summary_agent
from agents.router_agent import run_agent
import phoenix as px
import pandas as pd
import threading
from openinference.instrumentation import suppress_tracing
import Evaluation
import tracer

# Launch Phoenix Ui in background
def launch_phoenix():
     px.launch_app()

from sqlalchemy import create_engine

engine = create_engine(
    "sqlite:///phoenix.db",
    connect_args={"check_same_thread": False}
)

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join("outputs", timestamp)
os.makedirs(output_dir, exist_ok=True)



# help run streamlit app + phoenix tgr
if __name__ == "__main__":
    threading.Thread(target=launch_phoenix, daemon=True).start()

# Initialize Phoenix Client
client = px.Client()


# change Markdown format -> DataFrame
def parse_md_to_dataframe(md_path: str) -> pd.DataFrame:
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "## Summary" not in content:
        raise ValueError("Markdown file missing ## Summary section.")
    
    parts = content.split("## Summary")
    insights = parts[0].split("\n", 2)[-1].strip()
    summary = parts[1].strip()

    df = pd.DataFrame([{
        "insights": insights,
        "summary": summary
    }])

    st.subheader("Parsed DataFrame")
    st.dataframe(df)

    return df


# UI
st.title("Research Assistant Agent")

topic = st.text_input("Enter your research topic:")

if st.button("Generate Research Insights"):
    if topic:
        # Step 1: Router decides to call insight_agent
        st.info("Step 1: Router routing to Insight Agent...")
        from agents.router_agent import execute_tool_call
        router_result_1 = run_agent(topic)
        insights = execute_tool_call(router_result_1)

        # Step 2: Router decides to call summary_agent
        st.info("Step 2: Router routing to Summary Agent...")
        router_result_2 = run_agent(f"Please summarize these insights about {topic}: {insights}")
        summary = execute_tool_call(router_result_2)

        markdown = f"# Research Insights on {topic}\n\n{insights}\n\n## Summary\n\n{summary}"

        # Suppress tracing for file writing and parsing
        md_filename = f"Research_Insights_{topic.replace(' ', '_')}.md"
        md_path = os.path.join(output_dir, md_filename)

        with suppress_tracing():
            # write file
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown)

        st.success(f"Markdown file generated in folder: {output_dir}")

        st.download_button(
        "Download Markdown",
        data=markdown,
        file_name=md_filename
        )

        st.markdown(markdown)

        # parse the saved file
        df = parse_md_to_dataframe(md_path)

