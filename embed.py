"""
embed.py — Embed ingested JSONL chunks and load them into ChromaDB.

Uses sentence-transformers for embeddings (fully local, no API calls, and
independent of whatever model you're running in llama.cpp for generation).

Usage:
    python embed.py --input data/astro.jsonl --collection astro
    python embed.py --input data/rust.jsonl --collection rust

Requires (see requirements.txt):
    pip install chromadb sentence-transformers
"""

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# bge-small is a good default: small (~130MB), fast on CPU, strong retrieval
# quality for its size. Swap for "BAAI/bge-base-en-v1.5" if you want better
# quality and don't mind slower embedding / more RAM.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_and_load(
    input_path: str,
    collection_name: str,
    chroma_host: str,
    chroma_port: int,
    model_name: str,
    batch_size: int,
):
    chunks = load_chunks(input_path)
    if not chunks:
        print(f"No chunks found in {input_path}")
        return

    print(f"Loaded {len(chunks)} chunks from {input_path}")
    print(f"Loading embedding model: {model_name} (first run downloads it, then it's cached)")
    model = SentenceTransformer(model_name)

    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"embedding_model": model_name},
    )

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c["text"] for c in batch]

        # bge models expect no special prefix for documents (only queries
        # need "Represent this sentence for searching relevant passages: ")
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        ids = [f"{collection_name}-{start + i}" for i in range(len(batch))]
        metadatas = [
            {"url": c.get("url", ""), "heading": c.get("heading") or ""}
            for c in batch
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"  embedded + loaded {start + len(batch)}/{len(chunks)}")

    print(f"\nDone. Collection '{collection_name}' now has {collection.count()} chunks in Chroma.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed JSONL chunks and load them into ChromaDB.")
    parser.add_argument("--input", required=True, help="Path to the JSONL file from ingest.py")
    parser.add_argument("--collection", required=True, help="Chroma collection name, e.g. astro, rust")
    parser.add_argument("--chroma-host", default="chroma", help="Chroma host (service name inside Docker network)")
    parser.add_argument("--chroma-port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    embed_and_load(
        args.input,
        args.collection,
        args.chroma_host,
        args.chroma_port,
        args.model,
        args.batch_size,
    )
