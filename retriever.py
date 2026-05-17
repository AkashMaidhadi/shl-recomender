import json
import numpy as np
import faiss
import pickle
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
        """Use Gemini embedding API — free, no local model needed."""
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        return np.array(embeddings, dtype="float32")

    def _embed_query(self, query: str) -> np.ndarray:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        return np.array([result["embedding"]], dtype="float32")

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