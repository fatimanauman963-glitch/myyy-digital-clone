"""Run once on your laptop: builds chunks.json + embeddings.npy from data/."""
from rag import EMB_PATH, Retriever

r = Retriever(rebuild=True)
print(f"{len(r.chunks)} chunks embedded -> {EMB_PATH.name} {r.embeddings.shape}")
