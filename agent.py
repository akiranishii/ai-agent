import os
from mistralai import Mistral
import discord

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = "You are a helpful assistant."
MAX_HISTORY_LENGTH = 10  # Maximum number of messages to keep in history per channel


class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        # Dictionary to store conversation history for each channel
        self.channel_history = {}

    async def run(self, message: discord.Message):
        channel_id = message.channel.id
        
        # Initialize history for this channel if it doesn't exist
        if channel_id not in self.channel_history:
            self.channel_history[channel_id] = []
        
        # Add the current message to history
        self.channel_history[channel_id].append({"role": "user", "content": message.content})
        
        # Prepare messages including history and system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history (limited to MAX_HISTORY_LENGTH)
        history = self.channel_history[channel_id][-MAX_HISTORY_LENGTH:]
        messages.extend(history)
        
        # Get response from Mistral
        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )
        
        ai_response = response.choices[0].message.content
        
        # Add the assistant's response to the history
        self.channel_history[channel_id].append({"role": "assistant", "content": ai_response})
        
        # If history gets too long, trim it (keeping the most recent messages)
        if len(self.channel_history[channel_id]) > MAX_HISTORY_LENGTH * 2:
            self.channel_history[channel_id] = self.channel_history[channel_id][-MAX_HISTORY_LENGTH:]
        
        return ai_response

    async def run_with_text(self, text: str):
        """Process a single text input without maintaining conversation history"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
        
        # Get response from Mistral
        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )
        
        return response.choices[0].message.content
