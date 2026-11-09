"""
query.py — Ask a question against your ingested + embedded docs.

Embeds the query with the same model used in embed.py, retrieves the top-k
matching chunks from Chroma, and sends them as context to llama.cpp's
OpenAI-compatible chat endpoint.

Usage:
    python query.py --collection astro "how do I add a content collection?"
    python query.py --collection rust "what's the difference between String and &str?"

Requires (already in requirements.txt): chromadb, sentence-transformers, httpx
"""

import argparse

import chromadb
import httpx
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# bge models want this prefix on queries (but NOT on documents) for best results
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

SYSTEM_PROMPT_TEMPLATE = (
    "You are a tutor helping someone learn {topic}. "
    "Answer the question using ONLY the provided documentation excerpts below. "
    "If the excerpts don't contain enough information to answer, say so honestly "
    "rather than guessing. Include idiomatic code examples where relevant, and "
    "mention which section they came from.\n\n"
    "--- DOCUMENTATION EXCERPTS ---\n{context}\n--- END EXCERPTS ---"
)


def retrieve(question: str, collection_name: str, chroma_host: str, chroma_port: int, model_name: str, top_k: int):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([QUERY_PREFIX + question], normalize_embeddings=True).tolist()

    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_collection(name=collection_name)

    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "url": meta.get("url", ""), "heading": meta.get("heading", "")})
    return chunks


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        heading = f" ({c['heading']})" if c.get("heading") else ""
        parts.append(f"[{i}]{heading} {c['url']}\n{c['text']}")
    return "\n\n".join(parts)


def ask_llama_cpp(question: str, context: str, topic: str, llama_url: str):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(topic=topic, context=context)

    response = httpx.post(
        f"{llama_url}/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": 0.3,
            "max_tokens": 800,
        },
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Ask a question against your ingested docs.")
    parser.add_argument("question", help="Your question, in quotes")
    parser.add_argument("--collection", required=True, help="Chroma collection to search, e.g. astro, rust")
    parser.add_argument("--topic", default=None, help="Topic name for the tutor persona (defaults to --collection)")
    parser.add_argument("--chroma-host", default="chroma")
    parser.add_argument("--chroma-port", type=int, default=8000)
    parser.add_argument("--llama-url", default="http://host.docker.internal:8080", help="Base URL of the llama.cpp server (host.docker.internal reaches your Windows host's own llama-server from inside the container)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model (must match embed.py)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    topic = args.topic or args.collection

    print(f"Retrieving top {args.top_k} chunks for: {args.question!r}")
    chunks = retrieve(args.question, args.collection, args.chroma_host, args.chroma_port, args.model, args.top_k)

    if not chunks:
        print("No matching chunks found. Is the collection populated?")
        return

    print("Sources:")
    for c in chunks:
        print(f"  - {c['url']}" + (f"  ({c['heading']})" if c.get("heading") else ""))

    context = build_context(chunks)
    print("\nAsking llama.cpp...\n")
    answer = ask_llama_cpp(args.question, context, topic, args.llama_url)

    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
