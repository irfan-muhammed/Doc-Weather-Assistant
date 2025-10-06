# main.py
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Import the compiled graph from your agent.py file
from agent import app

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Weather & Document Assistant", layout="wide")
st.title("AI Weather & Document Assistant")
st.info("Ask about the weather or query the ISO 14229-1 document.")

# --- Session State for Chat History ---
# This will keep the conversation history in memory
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! How can I assist you today?")]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# --- Handle User Input and Run the Agent ---
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to session state and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show a spinner while the agent is thinking
    with st.spinner("Thinking..."):
        # Prepare the input for the LangGraph agent
        inputs = {
            "messages": [HumanMessage(content=prompt)],
            "query": prompt
        }
        
        # Invoke the agent to get the final response
        # We use .invoke() for the final result, not .stream()
        result = app.invoke(inputs)
        
        # Extract the final response from the agent's output
        final_response = result.get("response", "I'm sorry, I encountered an error.")

    # Add the agent's response to session state and display it
    st.session_state.messages.append(AIMessage(content=final_response))
    with st.chat_message("ai"):
        st.markdown(final_response)