import json
import numpy as np
import faiss
import pickle
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client

CATALOG_PATH = "catalog.json"
INDEX_PATH = "faiss.index"
META_PATH = "catalog_meta.pkl"


class CatalogRetriever:
    def __init__(self):
        self.assessments = []
        self.index = None

        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.assessments = pickle.load(f)
            print(f"Loaded index with {len(self.assessments)} assessments.")
        else:
            self._build_index()

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts using the new google-genai SDK."""
        embeddings = []
        for text in texts:
            response = get_client().models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            embeddings.append(response.embeddings[0].values)
        return np.array(embeddings, dtype="float32")

    def _embed_query(self, query: str) -> np.ndarray:
        response = get_client().models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return np.array([response.embeddings[0].values], dtype="float32")

    def _build_index(self):
        with open(CATALOG_PATH, "r") as f:
            self.assessments = json.load(f)

        texts = []
        for a in self.assessments:
            chunk = f"""
            Assessment: {a['name']}
            Test Types: {', '.join(a.get('test_types', []))}
            Remote Testing: {a.get('remote_testing', False)}
            Description: {a.get('description', '')}
            """.strip()
            texts.append(chunk)

        print("Embedding catalog using Gemini API...")
        embeddings = self._embed(texts)

        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.assessments, f) 

        print(f"Built and saved index with {len(self.assessments)} assessments.")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        query_vec = self._embed_query(query)
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
        name_set = {n.lower() for n in names}
        return [a for a in self.assessments if a["name"].lower() in name_set]