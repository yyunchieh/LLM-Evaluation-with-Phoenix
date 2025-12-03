# agents/insight_agent.py

from langchain_openai import ChatOpenAI
from tracer import tracer
from config import load_openai_api_key
from opentelemetry.trace import StatusCode #get_current_span
from utils.web_search import web_search_for_research

chat_model = ChatOpenAI(model="gpt-4o", temperature=0.5)

@tracer.chain()
def run_insight_agent(topic: str, use_web_search: bool = True) -> str:
    """
    Generate research insights on a topic

    Args:
        topic: Research topic
        use_web_search: Whether to use web search to ground insights (default: True)

    Returns:
        Generated insights in Markdown format
    """


    # Perform web search if enabled
    web_context = ""
    if use_web_search: #and "web_search" in version_info.get("features", []):
        print(f"   Performing web search for: {topic}")
        web_context = web_search_for_research(topic)
        print(f"   Web search completed, got {len(web_context)} characters of context")

    insights = []
    for i in range(5):
        # Adjust prompt based on whether we have web search context
        if web_context:
            prompt = f"""
        You are a highly knowledgeable research assistant specializing in {topic}.

        You have been provided with the following web search results about recent research:

        {web_context}

        Based on these search results and your knowledge, generate **structured research insights** in **Markdown format**.
        Your response should follow the exact structure below:

        # Research Insights on {topic}

        ## General Introduction
        Provide a **concise yet informative** introduction to {topic}.

        ## List of Research Papers
        Please cite **actual papers** from the search results above using **Chicago Style Citation format**.
        Only include papers that were mentioned in the search results.

        ## Future Research Directions
        Identify **novel research directions** that have not been widely explored.
        Focus on areas where **only a few studies exist** or where **new advancements are possible**.
        Provide at least **three unique ideas**, with a 1-2 sentence explanation for each.

        ## Title for Future Research

        Your response must be formatted in Markdown and strictly follow this structure.
        IMPORTANT: Base your insights on the search results provided above. Do not fabricate paper titles or citations.

        """
        else:
            prompt = f"""
        You are a highly knowledgeable research assistant specializing in {topic}.
        Your task is to generate **structured research insights** in **Markdown format**.
        Your response should follow the exact structure below:

        # Research Insights on {topic}

        ## General Introduction
        Provide a **concise yet informative** introduction to {topic}.

        ## List of Research Papers
        Please write in **Chicago Style Citation format**

        ## Future Research Directions
        Identify **novel research directions** that have not been widely explored.
        Focus on areas where **only a few studies exist** or where **new advancements are possible**.
        Provide at least **three unique ideas**, with a 1-2 sentence explanation for each.

        ## Title for Future Research

        Your response must be formatted in Markdown and strictly follow this structure.

        """

        try:
            response = chat_model.invoke(prompt)

            # Extract content from response
            content = response.content if hasattr(response, 'content') else str(response)
            insights.append(f" ## Response Set {i+1}\n\n{content}\n\n---\n")
        except Exception as e:
            insights.append(f"{e}")
    
    all_insights = "".join(insights)

    return all_insights