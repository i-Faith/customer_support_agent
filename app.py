# GRADIO INTERFACE
# Importing libraries
import gradio as gr
from dotenv import load_dotenv
import os


from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Loading the API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    print("Groq API Key loaded successfully!")
else:
    print("Error: Groq API Key not found. Check your .env file.")

# Creating the state class
class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

# Defining the LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# Testing the model
result = llm.invoke("Who are you?")
print(result.content)

# Keywords
technical_keywords = ["error", "bug", "crash", "technical issue", "problem", "won't work"]
billing_keywords = ["amount", "balance", "invoice", "refund", "payment", "charge", "billing"]
general_keywords = ["time", "deliver", "location", "info", "support", "contact"]

# Keyword route (helper function)
def keyword_route(query: str) -> str:
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in technical_keywords):
        return "Technical"
    elif any(keyword in query_lower for keyword in billing_keywords):
        return "Billing"
    elif any(keyword in query_lower for keyword in general_keywords):
        return "General"
    else:
        return "Unknown"

# Category function
def categorize(state: State) -> State:
    query = state["query"]
    category = keyword_route(query)
    if category == "Unknown":
        prompt = ChatPromptTemplate.from_template(
            "Categorize the following customer query into one of these categories: "
            "Technical, Billing, General. Query: {query}"
        )
        chain = prompt | llm
        category = chain.invoke({"query": query}).content
    return {"category": category}

# Sentiment function
def analyze_sentiment(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. "
        "Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {query}"
    )
    chain = prompt | llm
    sentiment = chain.invoke({"query": state["query"]}).content
    return {"sentiment": sentiment}

# Technical support
def handle_technical(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

# Billing support
def handle_billing(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

# General support
def handle_general(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a general support response to the following query: {query}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

# Escalate negative queries
def escalate(state: State) -> State:
    return {"response": "This query has been escalated to a human agent due to its negative sentiment"}

# Routing function
def route_query(state: State):
    keyword_category = keyword_route(state["query"])
    
    if state.get("sentiment") == "Negative":
        return "escalate"
    
    if keyword_category == "Technical":
        return "handle_technical"
    elif keyword_category == "Billing":
        return "handle_billing"
    elif keyword_category == "General":
        return "handle_general"
    
    # Fallback to LLM-based categorization
    category = state.get("category", "Unknown")
    if category == "Technical":
        return "handle_technical"
    elif category == "Billing":
        return "handle_billing"
    else:
        return "handle_general"

# Workflow
builder = StateGraph(State)
builder.add_node("categorize", categorize)
builder.add_node("analyze_sentiment", analyze_sentiment)
builder.add_node("handle_technical", handle_technical)
builder.add_node("handle_billing", handle_billing)
builder.add_node("handle_general", handle_general)
builder.add_node("escalate", escalate)

builder.add_edge("categorize", "analyze_sentiment")
builder.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "handle_technical": "handle_technical",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general",
        "escalate": "escalate"
    }
)
builder.add_edge("handle_technical", END)
builder.add_edge("handle_billing", END)
builder.add_edge("handle_general", END)
builder.add_edge("escalate", END)

builder.set_entry_point("categorize")
app = builder.compile()

# Customer support function
def run_customer_support(query: str) -> dict:
    result = app.invoke({"query": query})
    return {
        "category": result.get("category", "Unknown"),
        "sentiment": result.get("sentiment", "Neutral"),
        "response": result.get("response", "No response")
    }

# Gradio interface
def gradio_interface(query: str):
    result = run_customer_support(query)
    return (
        f"**Category:** {result['category']}\n"
        f"**Sentiment:** {result['sentiment']}\n"
        f"**Response:** {result['response']}"
    )

# Build the Gradio app
gui = gr.Interface(
    fn=gradio_interface,
    theme=gr.themes.Base(
        primary_hue="lime",    # yellow-green primary color
        secondary_hue="amber"  # warm secondary color
    ),
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs=gr.Markdown(),
    title="Customer Support Assistant",
    description="Provide a query and receive a categorized response. The system analyzes sentiment and routes to the appropriate support channel.",
)

gui.launch()
