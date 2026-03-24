<<<<<<< HEAD
Agentic RAG Project

This is a simple Agentic RAG system I built to understand how AI agents combine reasoning, retrieval, and tool usage.

The system can:

Solve math questions using a calculator tool

Retrieve relevant information from PDFs using RAG

Answer directly using Google Gemini

It uses FastAPI for the API, LangGraph for agent orchestration, SentenceTransformers for embeddings, and FAISS for vector similarity search.

How It Works

When a user sends a question, the agent first decides what to do:

If it contains math → use calculator

If it needs document knowledge → use RAG

Otherwise → answer directly with Gemini

If RAG is used, PDFs inside the data/ folder are split into chunks, converted into embeddings, stored in FAISS, and the top 3 relevant chunks are sent to Gemini to generate the final answer.

=======
# Agentic-RAG-AI-Assistant
>>>>>>> 7c0b1c6e83a249b2650f9489bd893464acd7fe38
