from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import google.generativeai as genai
from typing import List, Optional
import os

class Message(BaseModel):
    type: str
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Message] = []

class ChatResponse(BaseModel):
    response: str
    suggested_options: Optional[List[str]] = None

app = FastAPI(
    title="Fashion suggestion Chatbot API",
    description="API for interacting with a fashion chatbot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv('AIzaSyAlhkGue264_LOKUXakytcA2x5XacpwUuo', '')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def load_tiles_data() -> pd.DataFrame:
    """Load tile data from the CSV file"""
    try:
        df = pd.read_csv('final.csv')
        return df
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading CSV: {str(e)}")
print(load_tiles_data())
def generate_prompt_from_history(conversation_history: List[Message], df: pd.DataFrame) -> str:
    """Generate a prompt for the Gemini API based on the conversation history and Fashion data."""
    conversation_str = "\n".join([f"{msg.type}: {msg.content}" for msg in conversation_history])
    prompt = f"Conversation history:\n{conversation_str}\n\nHere are some outfit recommendations based on your preferences: {df.head(3)['OutfitName'].tolist()}.\nPlease provide a helpful fashion suggestion based on this information."

    return prompt

async def generate_response_from_gemini(prompt: str) -> str:
    """Generate a response from the Gemini AI model."""
    try:
        response = await genai.generate(
            model=model,
            prompt=prompt,
            temperature=0.7,
            max_output_tokens=200
        )
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Process a chat request and return a response from Gemini AI."""
    df = load_tiles_data()
    prompt = generate_prompt_from_history(request.conversation_history, df)
    response_text = await generate_response_from_gemini(prompt)
    suggested_options = df.head(3)['TileName'].tolist()
    return ChatResponse(response=response_text, suggested_options=suggested_options)

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    df = load_tiles_data()
    return {
        "status": "API is running",
        "data_loaded": not df.empty,
        "categories": [cat for cat in df['skin_tone'].unique() if pd.notna(cat)]
    }
