import faiss
import numpy as np

class EmbeddingStore:
    def __init__(self, dim=512):
        self.index = faiss.IndexFlatL2(dim)
        self.data = []

    def add(self, embedding, meta):
        self.index.add(np.array(embedding))
        self.data.append(meta)

    def search(self, query_embedding, top_k=3):
        D, I = self.index.search(np.array(query_embedding), top_k)
        return [self.data[i] for i in I[0]]