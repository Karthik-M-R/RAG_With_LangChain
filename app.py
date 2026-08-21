from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import os

if __name__ == "__main__":
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")

    if not os.path.exists("faiss_store/faiss.index"):
        print("[INFO] First run detected: building FAISS index from documents...")
        store.build_from_documents(docs)
    else:
        store.load()

    rag_search = RAGSearch()
    query = "What is DEFECTIVE PRODUCT / CONSUMER DISPUTE?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)