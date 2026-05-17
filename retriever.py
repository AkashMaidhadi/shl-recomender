import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os

MODEL_NAME = "all-MiniLM-L6-v2"  # fast, free, good quality
CATALOG_PATH = "catalog.json"
INDEX_PATH = "faiss.index"
META_PATH = "catalog_meta.pkl"

class CatalogRetriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.assessments = []
        self.index = None

        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            # Load pre-built index (fast startup on Render)
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.assessments = pickle.load(f)
            print(f"Loaded index with {len(self.assessments)} assessments.")
        else:
            self._build_index()

    def _build_index(self):
        with open(CATALOG_PATH, "r") as f:
            self.assessments = json.load(f)

        # Build a rich text chunk for each assessment to embed
        texts = []
        for a in self.assessments:
            chunk = f"""
            Assessment: {a['name']}
            Test Types: {', '.join(a.get('test_types', []))}
            Remote Testing: {a.get('remote_testing', False)}
            Description: {a.get('description', '')}
            """.strip()
            texts.append(chunk)

        print("Embedding catalog... this takes ~1 min first time.")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS flat index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner Product = cosine after normalization
        self.index.add(embeddings)

        # Save to disk so Render doesn't rebuild every cold start
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.assessments, f)

        print(f"Built and saved index with {len(self.assessments)} assessments.")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = self.assessments[idx].copy()
            item["score"] = float(score)
            results.append(item)

        return results

    def get_by_names(self, names: list[str]) -> list[dict]:
        """Fetch assessments by exact name — used for compare queries."""
        name_set = {n.lower() for n in names}
        return [a for a in self.assessments if a["name"].lower() in name_set]