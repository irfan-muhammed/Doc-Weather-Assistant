# AI Weather & Document Assistant

An agentic AI pipeline built with LangChain, LangGraph, and Gemini. This application can answer questions about the ISO 14229-1 technical document using RAG or provide real-time weather data for any city.



## Objective

This project was developed to demonstrate a comprehensive understanding of modern AI engineering principles, including Retrieval-Augmented Generation (RAG), agentic workflows, tool use, vector databases, and clean, tested code practices as per the assignment requirements.

## Features

-   **🧠 Agentic Routing**: A core router built with **LangGraph** intelligently decides whether to use the weather tool or the RAG pipeline based on the user's query.
-   **📄 RAG Pipeline**: Ingests the `ISO 14229-1:2013` PDF, chunks the content, generates embeddings using **Gemini**, and stores them in a **Qdrant** vector database for efficient retrieval.
-   **🌦️ Real-time Tool Use**: Fetches current weather information for any city by calling the **OpenWeatherMap API**.
-   **💬 Interactive UI**: A clean, user-friendly chat interface created with **Streamlit**.
-   **✅ Tested & Evaluated**: Includes unit tests with **Pytest** for key logic and is fully traceable with **LangSmith** for evaluation and debugging.

## Tech Stack

-   **Orchestration**: LangChain, LangGraph
-   **LLM & Embeddings**: Google Gemini (`gemini-1.5-flash`, `models/embedding-001`)
-   **Vector Database**: Qdrant Cloud
-   **UI**: Streamlit
-   **Tools**: OpenWeatherMap API
-   **Testing & Evaluation**: Pytest, Pytest-Mock, LangSmith

## Project Structure

```
.
├── .env
├── .gitignore
├── README.md
├── agent.py            # Core LangGraph agent logic (nodes, tools, graph assembly)
├── ingest.py           # Data ingestion script for parsing the PDF and populating Qdrant
├── main.py             # Streamlit UI application
├── requirements.txt
└── tests/
    ├── __init__.py
    ├── conftest.py     # Pytest configuration for pathing
    ├── test_agent.py   # Unit tests for the agent's router logic
    └── test_tools.py   # Unit tests for the weather API tool
```

---

## ⚙️ Setup and Installation

### 1. Clone the Repository

```bash
git clone <your-github-repo-url>
cd <repository-name>
```

### 2. Create a Virtual Environment and Install Dependencies

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a file named `.env` in the root of the project directory. Copy the contents of the example below and replace the placeholder values with your actual API keys.

```ini
# .env.example

# Google API Key for Gemini
GOOGLE_API_KEY="your_google_api_key"

# OpenWeatherMap API Key
OPENWEATHERMAP_API_KEY="your_openweathermap_api_key"

# Qdrant Cloud Connection Details
QDRANT_URL="your_qdrant_cloud_url"
QDRANT_API_KEY="your_qdrant_api_key"

# LangSmith Tracing (for evaluation)
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_PROJECT="Your-Project-Name"
```

---

## 🚀 Usage

### 1. Ingest the Data

Before running the main application, you must process the PDF document and populate your Qdrant vector database. Place your `ISO_14229-1_2013.pdf` file in the root directory.

```bash
python ingest.py
```
This script will parse the PDF, generate embeddings for the content, and upload them to your Qdrant collection.

### 2. Run the Streamlit Application

Launch the user interface with the following command:

```bash
streamlit run main.py
```
A new tab will open in your browser with the chat application.

---

## ✅ Testing

To ensure all components are working correctly, run the unit tests from the root directory:

```bash
pytest -v
```
All tests for the agent's router logic and the weather tool's API handling should pass.
