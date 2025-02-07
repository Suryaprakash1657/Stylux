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
    title="Fashion Suggestion Chatbot API",
    description="API for interacting with a fashion chatbot",
    version="1.0.0"
)

origins = ["*"]

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
    """Load and clean fashion data from the CSV file."""
    try:
        df = pd.read_csv('C:/Users/hp/Desktop/stfi/STYLUX-AI/backend/final.csv')

        # Remove unnecessary or empty columns
        df = df.drop(columns=['Unnamed: 6'], errors='ignore')

        # Drop rows with missing critical values
        df = df.dropna(subset=['skin_tone', 'recommended_outfit_(men)', 'why_this_outfit_(men)', 'shade', 'preferred_colors', 'style'])

        return df
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading CSV: {str(e)}")

def generate_prompt_from_history(conversation_history: List[Message], df: pd.DataFrame) -> str:
    """Generate a prompt for the AI model based on user history and fashion data."""
    
    # Extract user preferences from conversation history
    user_preferences = [msg.content for msg in conversation_history if msg.type == "user"]

    # Try to match the last mentioned preference (e.g., skin_tone or preferred color)
    if user_preferences:
        last_preference = user_preferences[-1].strip().lower()
        
        # Search dataset for matching skin tone or preferred color
        relevant_data = df[
            df['skin_tone'].str.lower().str.contains(last_preference, na=False) | 
            df['preferred_colors'].str.lower().str.contains(last_preference, na=False)
        ]
    else:
        relevant_data = df

    # Limit to top 3 relevant outfits
    outfits = relevant_data[['skin_tone', 'recommended_outfit_(men)', 'why_this_outfit_(men)', 'shade', 'preferred_colors', 'style']].head(3).to_dict(orient="records")

    # Format the output into a structured prompt
    outfit_details = "\n".join([
        f"Skin Tone: {item['skin_tone']}, Shade: {item['shade']}, Preferred Colors: {item['preferred_colors']}, "
        f"Style: {item['style']}, Outfit: {item['recommended_outfit_(men)']} - Why: {item['why_this_outfit_(men)']}"
        for item in outfits
    ])

    if not outfit_details:
        outfit_details = "No exact match found. Try specifying a different skin tone or color preference."

    # Build final prompt including conversation history
    conversation_str = "\n".join([f"{msg.type}: {msg.content}" for msg in conversation_history])

    final_prompt = f"""
    Conversation History:
    {conversation_str}

    Based on user preferences, here are some recommended outfits:
    {outfit_details}
    """

    return final_prompt.strip()


    # Build a dynamic prompt
    prompt = f"""
Conversation history:
{conversation_str}

Based on the fashion dataset:
{outfit_details}

Please provide a fashion suggestion considering the above details and user preferences.
"""
    return prompt

async def generate_response_from_gemini(prompt: str) -> str:
    """Generate a response from the Gemini AI model."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        df = load_fashion_data()
        prompt = generate_prompt_from_history(request.conversation_history, df)
        response_text = await generate_response_from_gemini(prompt)

        # Include both outfits and reasoning in the suggestions
        suggested_options = df[['recommended_outfit_(men)', 'why_this_outfit_(men)']].head(3).to_dict(orient="records")
        return ChatResponse(
            response=response_text,
            suggested_options=[f"{item['recommended_outfit_(men)']} - Why: {item['why_this_outfit_(men)']}" for item in suggested_options]
        )
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