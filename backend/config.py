import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Anthropic API settings
    ANTHROPIC_API_KEY: str = field(default="")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    
    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember
    
    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location
    
    def __post_init__(self):
        """Load environment variables after initialization"""
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    def reload(self):
        """Reload configuration from environment variables"""
        self.__post_init__()

config = Config()


