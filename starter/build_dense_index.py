"""Build dense embeddings index for all chunks."""
import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "chunks.json"
OUT_EMBEDDINGS = ROOT / "data" / "chunks_embeddings.json"

def main():
    print(f"Loading chunks from: {CHUNKS_PATH}")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"Found {len(chunks)} chunks.")

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Loading SentenceTransformer: {model_name}")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s.")

    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} chunk texts...")
    t1 = time.perf_counter()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    encode_time = time.perf_counter() - t1
    print(f"Encoded in {encode_time:.2f}s. Vector dimension: {vectors.shape[1]}")

    payload = []
    for c, vec in zip(chunks, vectors):
        payload.append({
            "id": c["id"],
            "embedding": vec.tolist()
        })

    OUT_EMBEDDINGS.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Saved {len(payload)} chunk embeddings to: {OUT_EMBEDDINGS}")

if __name__ == "__main__":
    main()
