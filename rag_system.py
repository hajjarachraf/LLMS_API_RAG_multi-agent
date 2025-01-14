import os
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

@dataclass
class RAGConfig:
    """Configuration for the RAG system"""
    db_name: str = "chroma_db_with_metadata"
    model_name: str = "gemini-1.5-pro"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retriever_k: int = 3
    retriever_threshold: float = 0.3

class CustomEmbeddings:
    """Custom embeddings class using sentence-transformers"""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

class RAGSystem:
    """RAG (Retrieval-Augmented Generation) system implementation"""
    
    DEFAULT_PROMPT = """Answer the question based on the following context. If you cannot find 
    the answer in the context, just say "I don't have enough information to answer that question."

    Context:
    {context}

    Question: {question}

    Answer: Let me help you with that."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.setup_environment()
        self.initialize_components()

    def setup_environment(self) -> None:
        """Set up environment variables and paths"""
        load_dotenv()
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        current_dir = Path(__file__).parent
        self.db_path = current_dir / "db" / self.config.db_name

    def initialize_components(self) -> None:
        """Initialize RAG system components"""
        # Initialize embeddings
        embeddings = CustomEmbeddings(self.config.embedding_model)
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=str(self.db_path),
            embedding_function=embeddings
        )
        
        # Initialize retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.config.retriever_k,
                "score_threshold": self.config.retriever_threshold
            }
        )
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model=self.config.model_name)
        
        # Initialize QA chain
        prompt = PromptTemplate(
            template=self.DEFAULT_PROMPT,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def get_response(self, question: str) -> Tuple[str, List[str]]:
        """Get response from RAG system"""
        try:
            result = self.qa_chain({"query": question})
            answer = result['result']
            sources = list(set(doc.metadata['source'] for doc in result['source_documents']))
            return answer, sources
        except Exception as e:
            return f"Error: {str(e)}", []

class ChatInterface:
    """Interactive chat interface for RAG system"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system

    def start(self) -> None:
        """Start the interactive chat loop"""
        print("\nWelcome to the RAG Chatbot! (Type 'exit' to quit)")
        print("=" * 50)
        
        while True:
            question = input("\nYou: ").strip()
            
            if question.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye!")
                break
                
            if not question:
                continue
                
            answer, sources = self.rag_system.get_response(question)
            
            print("\nAssistant:", answer)
            if sources:
                print("\nSources:", ", ".join(sources))
            print("\n" + "=" * 50)

def main():
    """Main entry point"""
    config = RAGConfig()
    rag_system = RAGSystem(config)
    chat_interface = ChatInterface(rag_system)
    chat_interface.start()

if __name__ == "__main__":
    main()