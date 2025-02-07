from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sys 
import google.generativeai as genai
from typing import List, Optional
from dotenv import load_dotenv
import os
app = FastAPI()


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
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv('GEMINI-API-KEY')
genai.configure(api_key='AIzaSyAlhkGue264_LOKUXakytcA2x5XacpwUuo')
model = genai.GenerativeModel('gemini-pro')

def load_fashion_data() -> pd.DataFrame:
    """Load fashion data from the CSV file"""
    try:
        df = pd.read_csv('C:/Users/hp/Desktop/stfi/STYLUX-AI/backend/final.csv')
        return df
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading CSV: {str(e)}")
print(load_fashion_data())
def generate_prompt_from_history(conversation_history: List[Message], df: pd.DataFrame) -> str:
    """Generate a prompt for the Gemini API based on the conversation history and Fashion data."""
    conversation_str = "\n".join([f"{msg.type}: {msg.content}" for msg in conversation_history])
    prompt = f"Conversation history:\n{conversation_str}\n\nHere are some outfit recommendations based on your preferences: {df.head(3)['recommended_outfit_(men)'].tolist()}.\nPlease provide a helpful fashion suggestion based on this information."

    return prompt

async def generate_response_from_gemini(prompt: str) -> str:
    """Generate a response from the Gemini AI model."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        # print(response)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        df = load_fashion_data()
        prompt = generate_prompt_from_history(request.conversation_history, df)
        response_text = await generate_response_from_gemini(prompt)
        suggested_options = df.head(3)['recommended_outfit_(men)'].tolist()
        return ChatResponse(response=response_text, suggested_options=suggested_options)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    df = load_fashion_data()
    return {
        "status": "API is running",
        "data_loaded": not df.empty,
        "categories": [cat for cat in df['skin_tone'].unique() if pd.notna(cat)]
    }