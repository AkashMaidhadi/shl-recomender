from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from agent import run_agent

app = FastAPI(title="SHL Assessment Recommender")

# Allow cross-origin requests (needed for web clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request / Response Schemas ---

class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Validate roles
    for m in request.messages:
        if m.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{m.role}'. Must be 'user' or 'assistant'."
            )

    # Last message must be from user
    if request.messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="Last message must be from the user."
        )

    # Enforce 8-turn cap (user + assistant turns combined)
    if len(request.messages) > 8:
        return ChatResponse(
            reply="We've reached the maximum conversation length. Please start a new session.",
            recommendations=[],
            end_of_conversation=True
        )

    # Convert Pydantic models to plain dicts for agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    result = run_agent(messages)

    return ChatResponse(
        reply=result["reply"],
        recommendations=result["recommendations"],
        end_of_conversation=result["end_of_conversation"]
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)