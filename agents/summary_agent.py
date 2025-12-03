# agents/summary_agent.py

from langchain_openai import ChatOpenAI
from tracer import tracer
from config import load_openai_api_key
from opentelemetry.trace import StatusCode, get_current_span


chat_model = ChatOpenAI(model="gpt-4o", temperature=0.5)

@tracer.chain()
def run_summary_agent(insights: str, topic: str) -> str:
    # Add experiment metadata to span
    span = get_current_span()

    prompt = f"""
    You are an expert researcher. Given the following five research insights on "{topic}", provide a **concise summary**.

    Your summary must be purely based on the provided research insights. **Do not introduce new information.**
    
    Extract 
    1.Key Points and Major Themes 
    2.Research Papers (list all of them that were mentioned) 
    3.Overall Trends 
    4.Future Research Directions 
    5.Suggested Title for Future Research
    

    {insights}

    Format the summary in **Markdown**.
    """
    
    response = chat_model.invoke(prompt)
    summary = response.content
    

    return summary