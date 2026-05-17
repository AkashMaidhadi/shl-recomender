# build_index.py  — run once locally before deploying
from retriever import CatalogRetriever
r = CatalogRetriever()  # triggers _build_index() since no index exists yet
print("Done! faiss.index and catalog_meta.pkl are ready.")