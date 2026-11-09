# Self-hosted docs RAG

A small, fully local RAG pipeline for learning a framework or language from its official docs. Crawl a docs site, embed it, and ask questions answered by a local llama.cpp model, with sources shown.

```
docs site -> ingest.py (Crawl4AI) -> data/<name>.jsonl
          -> embed.py (sentence-transformers) -> Chroma
question  -> query.py -> top-k chunks -> llama-server -> answer
```

Nothing is sent to a third-party API. The only network calls are fetching the docs you point it at, and a one-time download of the embedding model from Hugging Face.

## Requirements

- Docker (Docker Desktop on Windows/Mac)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) installed **on the host** (it is not run in Docker), plus a chat-capable `.gguf` model
- Optional: an NVIDIA GPU for fast generation (`-ngl 999`)

## Setup

1. Start llama.cpp on the host and leave it running:

   ```bash
   llama-server -m /path/to/model.gguf -c 8192 -ngl 999 --host 0.0.0.0 --port 8080
   ```

   Drop `-ngl 999` if you have no GPU. Use a larger `-c` if answers get cut off.

2. Build the images and start Chroma:

   ```bash
   docker compose build
   docker compose up -d chroma
   ```

## Usage

Run these from the project folder, one collection per docs site.

```bash
# 1. Crawl and chunk a docs site into data/astro.jsonl
docker compose run --rm ingest --source-url https://docs.astro.build --collection astro

# 2. Embed the chunks and load them into Chroma
docker compose run --rm embed --input data/astro.jsonl --collection astro

# 3. Ask a question
docker compose run --rm query --collection astro "how do I add a content collection?"
```

Swap the URL and collection name to add another site, e.g. `https://doc.rust-lang.org/book` with `--collection rust`.

## Configuration

| Setting | Where | Default |
|---|---|---|
| Chunks retrieved | `query.py --top-k` | 5 |
| Max answer length | `max_tokens` in `query.py` | 800 |
| llama.cpp URL | `query.py --llama-url` | `http://host.docker.internal:8080` |
| Embedding model | `--model` (embed and query must match) | `BAAI/bge-small-en-v1.5` |
| Pages crawled | `ingest.py --max-pages` | 300 |

## Notes and troubleshooting

- **`No address associated with hostname`**: the `query` container could not reach llama-server. Check that `llama-server` is running and that `curl http://localhost:8080/v1/models` works on the host.
- **Truncated answers**: raise `max_tokens` in `query.py` and/or `-c` on `llama-server`.
- **Code changes not taking effect**: scripts are baked into the image, so run `docker compose build` after editing them.
- **Heavy first build**: `sentence-transformers` pulls in PyTorch and Crawl4AI pulls in Chromium. Later builds reuse the cache.
- Ingestion assumes a static docs site, ideally with a `sitemap.xml`. Sites that fight scrapers may need a heavier tool.

## Layout

```
ingest.py            crawl + chunk by markdown header
embed.py             embed chunks, load into Chroma
query.py             retrieve + ask llama.cpp
Dockerfile           shared image for ingest/embed/query
docker-compose.yml   services: ingest, embed, query, chroma
```
