# tests/test_agent.py
import pytest
from agent import router_node
from langchain_core.messages import AIMessage

def test_router_node_routes_to_rag(mocker):
    """Tests if the router correctly decides 'rag' for a document-related query."""
    # Mock the llm.invoke method to return a predictable AIMessage
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value = AIMessage(content="rag")
    # This replaces the actual LLM in the agent.py file with our mock for this test
    mocker.patch("agent.llm", mock_llm)

    # Define a sample state
    state = {"query": "What is a Diagnostic Trouble Code?"}

    # Call the router node
    result = router_node(state)

    # Check if the decision is correct
    assert result == {"route": "rag"}

def test_router_node_routes_to_weather(mocker):
    """Tests if the router correctly decides 'weather' for a weather-related query."""
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value = AIMessage(content="weather")
    mocker.patch("agent.llm", mock_llm)

    state = {"query": "What is the weather in Bengaluru?"}
    result = router_node(state)
    assert result == {"route": "weather"}