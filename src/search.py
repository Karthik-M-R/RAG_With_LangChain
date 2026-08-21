import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from groq import BadRequestError

load_dotenv()

class RAGSearch:
    def __init__(self, vectorstore: FaissVectorStore = None):
        self.vectorstore = vectorstore or FaissVectorStore("faiss_store")
        self.vectorstore.load()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        try:
            self.llm = ChatGroq(
                model=self.model_name,
                groq_api_key=api_key,
                temperature=0.2,
            )
            print(f"[INFO] Groq LLM initialized: {self.model_name}")
        except BadRequestError:
            fallback_model = "llama-3.3-70b-versatile"
            self.llm = ChatGroq(
                model=fallback_model,
                groq_api_key=api_key,
                temperature=0.2,
            )
            print(f"[INFO] Groq LLM initialized with fallback model: {fallback_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)