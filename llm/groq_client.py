"""
Groq LLM Client Configuration
Provides Groq client for LLM-based query interpretation and output formatting.
NOT used in RL training - only for user interaction and output formatting.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    print("Warning: groq package not installed. LLM features will use fallback mode.")
    print("Install with: pip install groq")


class GroqClient:
    """
    Groq LLM client wrapper.
    Used ONLY for query interpretation and output formatting.
    NOT used in RL training or environment dynamics.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = None
        self.available = False
        
        if HAS_GROQ and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                self.available = True
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {e}")
                self.available = False
        else:
            if not HAS_GROQ:
                print("Warning: groq package not installed")
            if not self.api_key:
                print("Warning: GROQ_API_KEY not set in environment")
    
    def call_llm(
        self,
        prompt: str,
        model: str = "llama3-70b-8192",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Call Groq LLM with a prompt.
        
        Args:
            prompt: Input prompt
            model: Model to use (llama3-70b-8192, mixtral-8x7b-32768, gemma2-27b-it)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            LLM response text or None if failed
        """
        if not self.available or not self.client:
            return None
        
        # Model priority list (try best available)
        # Updated: llama3-70b-8192 is decommissioned, using newer models
        model_priority = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-27b-it"
        ]
        
        if model not in model_priority:
            model = model_priority[0]  # Use latest available model
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error calling Groq LLM: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Groq client is available."""
        return self.available


# Global client instance
_groq_client = None

def get_groq_client() -> GroqClient:
    """Get or create global Groq client instance."""
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client

