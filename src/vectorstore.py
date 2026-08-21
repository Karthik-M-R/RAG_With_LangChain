import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def _ensure_index(self, dim: int = None):
        if self.index is not None:
            return
        if dim is None:
            dim = self.model.encode(["test"]).astype("float32").shape[1]
        self.index = faiss.IndexFlatL2(dim)
        print(f"[INFO] Created a new FAISS index with dimension {dim}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        if not chunks:
            print("[INFO] No chunks created from documents.")
            return

        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.size == 0:
            print("[INFO] No embeddings to add.")
            return

        self._ensure_index(embeddings.shape[1])
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors to FAISS index.")

    def save(self):
        if self.index is None:
            print("[INFO] No FAISS index to save; skipping.")
            return

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved FAISS index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        if not os.path.exists(faiss_path):
            print(f"[INFO] No FAISS index found at {faiss_path}; creating a new empty index.")
            self.metadata = []
            self._ensure_index()
            return

        self.index = faiss.read_index(faiss_path)
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = []
        print(f"[INFO] Loaded FAISS index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            raise ValueError("Vector store is not initialized. Call load() or build_from_documents() first.")

        query_embedding = np.asarray(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        if self.index is None:
            self.load()
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)