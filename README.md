# NPC-Chatbot

**Agentic Hybrid RAG · LLM · EHR Visualization**

An intelligent Non-Player Character (NPC) chatbot that combines Agentic AI, Hybrid Retrieval-Augmented Generation (RAG), and Large Language Models (LLM) to deliver dynamic, context-aware conversations — with integrated Electronic Health Record (EHR) visualization for clinical or simulation use cases.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

NPC-Chatbot is designed to simulate realistic, knowledge-grounded conversations with virtual characters (NPCs) in games, simulations, or clinical training environments. It leverages:

- **Agentic AI** – The chatbot can autonomously plan, reason, and take multi-step actions to answer complex queries.
- **Hybrid RAG** – Combines dense vector search and sparse keyword retrieval to fetch the most relevant context before generating a response.
- **LLM backbone** – A large language model produces fluent, coherent, and contextually accurate dialogue.
- **EHR Visualization** – Renders Electronic Health Record data (patient history, vitals, medications, etc.) alongside chat responses for clinical simulation and medical training scenarios.

---

## Features

- 🤖 **Agentic reasoning** — multi-step planning and tool use for complex queries
- 🔍 **Hybrid RAG retrieval** — combines vector similarity search with BM25 keyword search for high-quality context grounding
- 💬 **LLM-powered dialogue** — natural, context-aware NPC conversation
- 🏥 **EHR visualization** — interactive display of patient records, vitals, lab results, and clinical notes
- 🧠 **Memory & context management** — tracks conversation history for coherent long-form dialogue
- ⚙️ **Configurable personas** — define NPC backstory, knowledge scope, and personality through configuration files

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   User / Client                  │
└───────────────────────┬─────────────────────────┘
                        │ Chat Request
                        ▼
┌─────────────────────────────────────────────────┐
│               Agentic Orchestrator               │
│  • Intent classification                         │
│  • Multi-step reasoning & tool dispatch          │
└───────┬───────────────┬─────────────────────────┘
        │               │
        ▼               ▼
┌───────────────┐ ┌─────────────────────────────┐
│  Hybrid RAG   │ │       EHR Data Layer         │
│  Retriever    │ │  • Patient records           │
│  • Dense (vec)│ │  • Vitals / Labs             │
│  • Sparse(BM25)│ │  • Medications / Notes       │
└───────┬───────┘ └──────────────┬───────────────┘
        │                        │
        └──────────┬─────────────┘
                   ▼
        ┌──────────────────────┐
        │     LLM (Generator)   │
        │  Generates response   │
        │  grounded in context  │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Response + EHR     │
        │   Visualization UI   │
        └──────────────────────┘
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher (or the runtime required by the project)
- An LLM API key (e.g., OpenAI, Azure OpenAI, or a local model server)
- A vector database (e.g., Chroma, Qdrant, or Weaviate) for RAG retrieval
- Node.js (if a web-based visualization frontend is included)

### Installation

```bash
# Clone the repository
git clone https://github.com/planktong/NPC-Chatbot.git
cd NPC-Chatbot

# Install Python dependencies
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Description |
|---|---|
| `LLM_API_KEY` | API key for the LLM provider |
| `LLM_MODEL` | Model name (e.g., `gpt-4o`) |
| `VECTOR_DB_URL` | Connection URL for the vector database |
| `EHR_DATA_PATH` | Path to EHR data files or database connection string |

### Running the Application

```bash
# Start the chatbot server
python main.py

# (Optional) Start the EHR visualization frontend
npm install && npm start
```

---

## Usage

Once the application is running, open the chat interface and interact with the NPC:

```
You: What medications is the patient currently taking?

NPC: Based on the patient record, they are currently prescribed
     Metformin 500mg twice daily and Lisinopril 10mg once daily.
     [EHR panel displays medication list with dosage history]
```

The EHR visualization panel updates in real time alongside the conversation, surfacing relevant clinical data as the dialogue progresses.

---

## Project Structure

```
NPC-Chatbot/
├── agents/           # Agentic orchestration and tool definitions
├── rag/              # Hybrid RAG retriever (dense + sparse)
├── llm/              # LLM integration and prompt templates
├── ehr/              # EHR data ingestion and visualization
├── ui/               # Frontend visualization components
├── config/           # NPC personas and application settings
├── data/             # Sample EHR datasets and knowledge bases
├── tests/            # Unit and integration tests
├── main.py           # Application entry point
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the [MIT License](LICENSE).
