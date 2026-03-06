from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

"""FastAPI application entry point.

When running via `uvicorn app:app` the module is executed as a top-level
script, so imports should be relative to the working directory rather than a
package. Using bare imports keeps the code runnable both as a module and from
a virtual environment shell.
"""

from graph import run_agent
from rag import get_relevant_docs

app = FastAPI(title="Agentic RAG API")


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: QuestionRequest) -> Dict[str, Any]:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = run_agent(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # retrieval docs if any
    docs = []
    try:
        docs = get_relevant_docs(question)
    except Exception:
        # ignore retrieval failure, we'll report empty
        docs = []

    return {
        "answer": result.answer,
        "tool_used": result.tool_used,
        "retrieved_docs": docs,
    }
