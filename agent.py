# agent.py
import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

# --- 1. Load Environment Variables ---
load_dotenv()

# Check if necessary API keys are set
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("Google API key not found in .env file")
if "OPENWEATHERMAP_API_KEY" not in os.environ:
    raise ValueError("OpenWeatherMap API key not found in .env file")

# --- 2. Define the State for our Graph ---

# The state is a dictionary that will be passed between nodes.
# It holds the information that our agent needs to do its work.
class AgentState(TypedDict):
    query: str  # The user's input query
    messages: Annotated[List[BaseMessage], lambda x, y: x + y] # The chat history
    response: str # The final response to the user
    sender: str # To track who is speaking (user or agent)
    documents: List[dict] # To hold retrieved documents for RAG
    weather_data: dict | None # <-- ADD THIS LINE
    route: str # <-- ADD THIS LINE

import requests
from langchain_qdrant import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
# --- 3. Define Tools ---

# Tool for fetching weather
class WeatherInput(BaseModel):
    city: str = Field(description="The city to get the weather for.")

def fetch_weather(city: str) -> dict:
    """Fetches real-time weather data for a given city."""
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": os.getenv("OPENWEATHERMAP_API_KEY"),
        "units": "metric"  # Use Celsius
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            return {"error": f"City '{city}' not found."}
        else:
            return {"error": f"HTTP error occurred: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"An error occurred: {req_err}"}

# Tool for RAG
# First, set up the necessary components for our retriever
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = "iso14229_uds_pages"

# Initialize the Qdrant vector store
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

# 2. Initialize the LangChain Qdrant wrapper
qdrant_store = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings,
)

# Create the retriever
retriever = qdrant_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # Retrieve the top 5 most similar documents
)

def get_rag_retriever():
    """Returns the configured Qdrant retriever for RAG."""
    return retriever

# --- 4. Define Graph Nodes ---

def router_node(state: AgentState) -> dict:
    """
    Decides the next step and updates the 'route' field in the state.
    """
    print("---ROUTER ---")
    query = state["query"]
    prompt = f"""Given the user query, should I use the 'weather_tool' for real-time weather information 
    or the 'rag_tool' for answering questions based on the provided ISO 14229-1 document?

    User Query: "{query}"

    Respond with only 'weather' or 'rag'.
    """

    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    print(f"Router decision: '{decision}'")

    if "weather" in decision:
        return {"route": "weather"}
    else:
        return {"route": "rag"}

def rag_node(state: AgentState) -> dict:
    """
    Retrieves documents from the Qdrant database based on the query.
    """
    print("---RETRIEVING FROM RAG ---")
    query = state["query"]
    docs = retriever.invoke(query)
    
    # Format documents for clarity in the final prompt
    documents_content = [doc.page_content for doc in docs]
    print(f"Retrieved {len(documents_content)} documents.")
    
    return {"documents": documents_content, "sender": "RAG Tool"}

def weather_node(state: AgentState) -> dict:
    """
    Extracts the city from the query and fetches weather data.
    """
    print("---FETCHING WEATHER ---")
    query = state["query"]
    
    # Use the LLM with a structured output to reliably get the city name
    structured_llm = llm.with_structured_output(WeatherInput)
    result = structured_llm.invoke(query)
    
    if not result.city:
        return {"weather_data": {"error": "Could not identify a city in the query."}, "sender": "Weather Tool"}

    print(f"City identified: {result.city}")
    weather_info = fetch_weather(result.city)
    
    return {"weather_data": weather_info, "sender": "Weather Tool"}

def response_generator_node(state: AgentState) -> dict:
    """
    Generates a final, human-readable response based on the gathered information.
    """
    print("---GENERATING RESPONSE ---")
    query = state["query"]
    documents = state.get("documents")
    weather_data = state.get("weather_data")

    if documents:
        # Generate response using RAG context
        context = "\n\n".join(documents)
        prompt = f"""Based on the following context from the ISO 14229-1 document, 
        provide a concise answer to the user's question.

        Context:
        {context}

        Question:
        {query}
        """
        response = llm.invoke(prompt)
        return {"response": response.content, "sender": "Agent"}
    
    elif weather_data:
        # Generate response using weather data
        prompt = f"""The user asked: "{query}"
        Here is the real-time weather data: {weather_data}.
        
        Generate a friendly, conversational response summarizing this weather information.
        If there is an error in the data, state it clearly.
        """
        response = llm.invoke(prompt)
        return {"response": response.content, "sender": "Agent"}
    
    else:
        # Fallback response
        return {"response": "I'm sorry, I couldn't process that request.", "sender": "Agent"}

from langgraph.graph import StateGraph, END

# --- 5. Assemble the Graph ---

# Create a new graph
workflow = StateGraph(AgentState)

# Add the nodes to the graph
workflow.add_node("router", router_node)
workflow.add_node("rag_tool", rag_node)
workflow.add_node("weather_tool", weather_node)
workflow.add_node("responder", response_generator_node)

# Set the entry point for the graph
workflow.set_entry_point("router")

# Define the conditional edges
# This tells the graph how to move from the router to the correct tool
workflow.add_conditional_edges(
    "router",
    # The lambda function now inspects the 'route' field in the state
    lambda state: state["route"],
    {
        "rag": "rag_tool",
        "weather": "weather_tool",
    },)
# Define the normal edges
# After a tool is called, the flow always goes to the responder
workflow.add_edge("rag_tool", "responder")
workflow.add_edge("weather_tool", "responder")

# The responder is the final step, so we connect it to the END
workflow.add_edge("responder", END)

# Compile the graph into a runnable app
app = workflow.compile()

# --- (Optional) Code to test the agent directly ---
# --- (Optional) Code to test the agent directly ---
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    # --- Example 1: RAG Query ---
    rag_query = "what is Diagnostic Trouble Code?"
    inputs = {
        "messages": [HumanMessage(content=rag_query)],
        "query": rag_query  # <-- ADD THIS KEY
    }
    
    print("---Testing RAG Query ---")
    for output in app.stream(inputs, stream_mode="values"):
        print(output)
        print("---")
    
    print("" + "="*30 + "")

    # --- Example 2: Weather Query ---
    weather_query = "what is the weather like in Bengaluru?"
    inputs = {
        "messages": [HumanMessage(content=weather_query)],
        "query": weather_query # <-- ADD THIS KEY
    }

    print("---Testing Weather Query ---")
    for output in app.stream(inputs, stream_mode="values"):
        print(output)
        print("---")