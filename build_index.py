import os

# Remove old index files so retriever rebuilds fresh
for f in ["faiss.index", "catalog_meta.pkl"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed old {f}")

from retriever import CatalogRetriever
r = CatalogRetriever()
print("Done! faiss.index and catalog_meta.pkl are ready.")