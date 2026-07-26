import os
from typing import TypedDict, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    code_diff: str               # The raw code changes pulled from GitHub
    security_review: str        # Output from the Security Agent
    performance_review: str     # Output from the Performance Agent
    style_review: str           # Output from the Style Agent
    final_report: str           # The compiled, clean report for the PR comment

llm = ChatOpenAI(
    model="pocohere/north-mini-code:free",
    api_key=os.getenv("OPENROUTER_API_KEY"), 
    base_url="https://openrouter.ai/api/v1",
    streaming=False,
)

def security_agent(state: AgentState) -> Dict[str, Any]:
    """Inspects code diffs exclusively for vulnerabilities and credential leaks."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert Application Security Engineer (AppSec).\n"
            "Review the following Git diff strictly for security flaws: hardcoded secrets/keys, "
            "SQL injections, XSS, insecure dependencies, or broken access control.\n"
            "Provide your findings in clear bullet points. If no security issues are found, "
            "explicitly state 'No critical security issues identified.'"
        )),
        ("human", "Analyze this diff:\n\n{diff}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"diff": state["code_diff"]})
    return {"security_review": response.content}


def performance_agent(state: AgentState) -> Dict[str, Any]:
    """Inspects code diffs for runtime execution speed, memory leaks, and scaling blocks."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Principal Software Engineer focusing on performance engineering.\n"
            "Review the following Git diff for performance bottlenecks: unoptimized loops, "
            "redundant database queries (N+1 problems), memory leaks, or heavy synchronous blocks.\n"
            "Provide tactical optimization feedback. If performance is fine, state 'Performance looks solid.'"
        )),
        ("human", "Analyze this diff:\n\n{diff}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"diff": state["code_diff"]})
    return {"performance_review": response.content}


def style_agent(state: AgentState) -> Dict[str, Any]:
    """Inspects code diffs for readability, design anti-patterns, and architecture."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Senior Staff Engineer focused on clean code, readability, and design patterns.\n"
            "Review the following Git diff for maintainability, poor variable naming, lack of comments, "
            "or architectural inconsistencies.\n"
            "Keep feedback concise. If clean, state 'Code style looks excellent.'"
        )),
        ("human", "Analyze this diff:\n\n{diff}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"diff": state["code_diff"]})
    return {"style_review": response.content}


def orchestrator_agent(state: AgentState) -> Dict[str, Any]:
    """Merges all agent outputs, deduplicates findings, and creates a beautiful PR comment."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the Lead PR Orchestrator. Your job is to compile the reviews from your Security, "
            "Performance, and Style specialists into a single, cohesive, polite GitHub PR comment.\n"
            "Deduplicate any overlapping points. Format the output professionally using crisp Markdown "
            "headers, tables, or quote blocks. Make it easy for the developer to skim."
        )),
        ("human", (
            "Here are the individual reviews:\n\n"
            "### Security Feedback:\n{security}\n\n"
            "### Performance Feedback:\n{performance}\n\n"
            "### Style Feedback:\n{style}\n\n"
            "Compile them into a single, structured markdown comment."
        ))
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "security": state["security_review"],
        "performance": state["performance_review"],
        "style": state["style_review"]
    })
    return {"final_report": response.content}

workflow = StateGraph(AgentState)

retry_in_all_cases = RetryPolicy(
    max_attempts=5,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=120,
    retry_on=Exception
)

workflow.add_node("security_node", security_agent, retry_policy=retry_in_all_cases)
workflow.add_node("performance_node", performance_agent, retry_policy=retry_in_all_cases)
workflow.add_node("style_node", style_agent, retry_policy=retry_in_all_cases)
workflow.add_node("orchestrator_node", orchestrator_agent, retry_policy=retry_in_all_cases)

workflow.set_entry_point("security_node")
workflow.set_entry_point("performance_node")
workflow.set_entry_point("style_node")

workflow.add_edge("security_node", "orchestrator_node")
workflow.add_edge("performance_node", "orchestrator_node")
workflow.add_edge("style_node", "orchestrator_node")

workflow.add_edge("orchestrator_node", END)

app = workflow.compile()

if __name__ == "__main__":
    # A mock diff featuring: a hardcoded secret, a SQL injection, and a nested loop bottleneck
    mock_diff = """
    diff --git a/app/users.py b/app/users.py
    index 8372hdb..9283jhd 100644
    --- a/app/users.py
    +++ b/app/users.py
    @@ -10,4 +10,18 @@ def get_user_data(user_id):
    +    # Fix later: staging key
    +    API_SECRET_KEY = "sk_live_51NxF2...secret_string"
    +    
    +    # Querying database directly from input string
    +    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    +    db.execute(query)
    +
    +    # Calculate matrix lookup for user history
    +    for i in range(len(history)):
    +        for j in range(len(history)):
    +            for k in range(len(history)):
    +                process_matrix(i, j, k)
    """

    print("Triggering Multi-Agent Code Review Pipeline...")
    
    inputs = {"code_diff": mock_diff}
    output = app.invoke(inputs)
    
    print("\n--- Final Generated GitHub PR Comment ---\n")
    print(output["final_report"])