# AI Study Assistant

[![CI](https://github.com/JesseSaarinen/AI-study-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/JesseSaarinen/AI-study-assistant/actions/workflows/ci.yml)
[![Retrieval evaluation](https://github.com/JesseSaarinen/AI-study-assistant/actions/workflows/evaluation.yml/badge.svg)](https://github.com/JesseSaarinen/AI-study-assistant/actions/workflows/evaluation.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-study-assistant-jesse.streamlit.app)

Ask questions about your own lecture notes and get answers that cite the exact PDF page they came from. When the notes do not contain the answer, the assistant says so instead of guessing.

It is a retrieval-augmented generation (RAG) app built with Streamlit, ChromaDB, FastEmbed and any OpenAI-compatible chat API, including free tiers from Gemini and Groq.

**[Try the live demo](https://ai-study-assistant-jesse.streamlit.app)**: click **Try with sample notes**, then pick an example question. No account or API key needed. The app may take a minute to wake up if nobody has used it recently.

![Answer with citations](docs/AI-assistant-answer-example.png)

## Highlights

- **Grounded answers with page citations.** Answers cite `[Source N]` markers, and the UI shows the file, page and retrieved text behind each one.
- **Says "I could not find this in the uploaded notes."** when retrieval finds nothing relevant, without calling the chat model at all.
- **Measured, not guessed.** A labelled question set and an evaluation script compare embedding models, chunk sizes and distance thresholds, and the results set the app's defaults. [Results below](#retrieval-evaluation).
- **Light enough for free hosting.** Embeddings run on ONNX Runtime through FastEmbed, so the install is about 1 GB instead of 6.6 GB with PyTorch.
- **Ready for a public demo.** Bundled sample notes, one-click example questions, and a demo mode that isolates each visitor's notes and caps API usage.
- **Tested.** 119 pytest tests cover every module, the full pipeline and the Streamlit UI, with no network calls. CI runs ruff and the tests on every pull request and every push to `main`.

## How it works

```mermaid
flowchart LR
    subgraph Ingestion
        A[PDF upload] --> B[PyMuPDF<br/>page text]
        B --> C[Page-aware<br/>chunking]
        C --> D[FastEmbed<br/>all-MiniLM-L6-v2]
        D --> E[(ChromaDB<br/>cosine distance)]
    end

    subgraph Question answering
        Q[Question] --> QE[Embed question]
        QE --> R[Top-k retrieval]
        E --> R
        R --> F{Any chunk under<br/>distance threshold?}
        F -- no --> NF[Not found in notes]
        F -- yes --> P[Grounded prompt<br/>with Source labels]
        P --> L[OpenAI-compatible<br/>chat model]
        L --> AN[Answer with<br/>citations]
    end
```

1. **Extract.** PyMuPDF reads the selectable text of each page. Empty, oversized, non-PDF and scanned files are reported per file, so one bad upload does not stop the rest.
2. **Chunk.** Each page is split on its own into overlapping chunks that break at paragraph, sentence or word boundaries. Chunks never cross pages, so every chunk has exactly one page number to cite.
3. **Embed and store.** Chunks are embedded with `all-MiniLM-L6-v2`, normalised, and stored in ChromaDB with their file name and page number.
4. **Retrieve.** The question is embedded the same way and the closest chunks are fetched. Chunks above `RETRIEVAL_DISTANCE_THRESHOLD` are dropped.
5. **Answer.** The remaining chunks go into a prompt that labels them `[Source 1]`, `[Source 2]` and so on, and tells the model to answer only from them. If the model forgets to cite, the retrieved source labels are appended to the answer.

## Quick start

Requires Python 3.11 or newer.

```bash
git clone https://github.com/JesseSaarinen/AI-study-assistant.git
cd AI-study-assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
cp .env.example .env             # Windows: Copy-Item .env.example .env
```

Add an API key to `.env` (the file is git-ignored), then start the app:

```bash
streamlit run app.py
```

Open http://localhost:8501 and click **Try with sample notes**, or upload your own PDFs.

### Choosing a chat provider

The chat model is reached through the OpenAI Python client, so any OpenAI-compatible API works. Set these three values in `.env`:

| Provider | `OPENAI_BASE_URL` | Example `OPENAI_MODEL` | Cost |
|---|---|---|---|
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-3.5-flash-lite` | Free tier |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` | Free tier |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.2` | Free, runs offline |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | Paid |

For Ollama, set `OPENAI_API_KEY` to any non-empty value such as `ollama`. Model names change often, so check your provider's current list.

## Using the app

The app has three tabs that follow the pipeline:

1. **Process notes.** Upload PDFs and process them end to end, or load the bundled sample notes. An advanced panel runs each stage on its own, which is handy for debugging or for showing how RAG works.
2. **Inspect data.** Browse the extracted pages, chunks, embedding vectors and the records stored in ChromaDB.
3. **Ask questions.** Ask about the stored notes. Each answer lists its citations and the retrieved chunks with their distance scores, with the cited ones marked.

![Upload and processing](docs/AI-assistant-uploadfile-example.png)

## Retrieval evaluation

[`scripts/evaluate_retrieval.py`](scripts/evaluate_retrieval.py) indexes the bundled sample notes and runs 38 labelled questions from [`eval/questions.json`](eval/questions.json): 30 that the notes answer, each labelled with the page that answers it, and 8 they cannot answer, including near misses such as B-trees and A* search. No API key is needed. The evaluation runs in GitHub Actions on every pull request that touches the pipeline, and the check fails if Hit@1 for the default setup drops below 90%. Full results are in [`eval/results.md`](eval/results.md).

| Embedding model | Chunk size | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` (default) | 1000 | 97% | 100% | 0.98 |
| `all-MiniLM-L6-v2` | 1500 | 87% | 100% | 0.92 |
| `BAAI/bge-small-en-v1.5` | 1000 | 90% | 97% | 0.94 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 1000 | 67% | 83% | 0.75 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 300 | 83% | 97% | 0.89 |

Findings:

- **The default setup ranks the right page first for 29 of 30 questions.** Chunks of 300 to 1000 characters perform the same; at 1500, unrelated topics start sharing a chunk and Hit@1 drops to 87%.
- **The old distance threshold filtered almost nothing.** At 1.0 only 1 of the 8 unanswerable questions was rejected before reaching the chat model. At 0.8 every answerable question still finds its page and 3 of 8 are rejected; at 0.7, 6 of 8 are rejected at the cost of one answerable question. The default is now 0.8, a deliberately cautious choice because real notes are messier than this test set, and the prompt still tells the model to say when the context does not contain the answer.
- **Thresholds do not transfer between models.** `bge-small-en-v1.5` places every chunk within 0.5 of the question, so a threshold of 0.8 filters nothing for it.
- **The multilingual model is weaker on English notes and prefers small chunks.** If you use it for Finnish notes, try a `CHUNK_SIZE` of 300 to 500.

## Design decisions

- **Chunks never cross page boundaries.** It costs a little context at page edges, but every chunk maps to one page, so a citation always points to a page the student can open.
- **Cosine distance on normalised vectors.** Distances stay between 0 and 2 regardless of text length, which makes a single threshold meaningful across documents.
- **Filter before the model, not after.** If no chunk is close enough, the app returns the not-found message without an API call. That is cheaper, and it removes the main source of hallucinated answers: weakly related context.
- **FastEmbed instead of Sentence Transformers.** The same model runs on ONNX Runtime, which removes PyTorch and its CUDA libraries from the install. Swapping models is one setting, and the evaluation script shows what each choice costs.
- **Any OpenAI-compatible API.** Free tiers and local models work with no code changes, which matters for students without a paid API account.
- **`src/` has no Streamlit imports.** The pipeline is plain Python, so the UI, the tests and the evaluation script all run the same code. Streamlit code lives in `ui/`, one module per tab.
- **Every model is behind a small protocol.** The embedding model and chat model are typed as minimal interfaces, so tests use deterministic fakes and never touch the network.

## Project structure

```text
AI-study-assistant/
├── app.py                  # Streamlit entry point
├── src/                    # RAG pipeline, no Streamlit imports
│   ├── pdf_loader.py       # PyMuPDF text extraction and validation
│   ├── chunker.py          # overlapping, page-aware text chunks
│   ├── embeddings.py       # FastEmbed (ONNX) embedding model wrapper
│   ├── vector_store.py     # ChromaDB storage and retrieval
│   ├── ingestion.py        # extract → chunk → embed → store
│   ├── prompts.py          # grounded system and user prompts
│   ├── rag_pipeline.py     # retrieval and answer generation
│   ├── citations.py        # [Source N] parsing and formatting
│   ├── evaluation.py       # hit rate, MRR and threshold metrics
│   ├── config.py           # environment-based settings with validation
│   ├── models.py           # dataclasses shared across modules
│   └── ui_helpers.py       # pure formatting helpers for the UI
├── ui/                     # Streamlit components, one module per tab
├── tests/                  # one test module per source module, plus app tests
├── eval/                   # labelled questions and the latest results
├── scripts/                # sample notes builder and evaluation script
├── sample_data/            # bundled sample lecture notes (PDF)
├── docs/                   # deployment guide and screenshots
├── .github/workflows/      # CI and retrieval evaluation
├── requirements.txt        # runtime dependencies
├── requirements-dev.txt    # adds pytest and ruff
└── pyproject.toml          # pytest, coverage and ruff settings
```

## Configuration

All settings are environment variables, read from `.env` locally or from Streamlit secrets when hosted. See [`.env.example`](.env.example) for a commented template.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | | API key for the chat model |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model name |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Embedding model, any model FastEmbed supports |
| `CHUNK_SIZE` | `1000` | Maximum chunk length in characters |
| `CHUNK_OVERLAP` | `150` | Characters shared by neighbouring chunks |
| `TOP_K` | `5` | Chunks retrieved per question |
| `RETRIEVAL_DISTANCE_THRESHOLD` | `0.8` | Maximum cosine distance, or `disabled`. Tuned per model, see [results](eval/results.md) |
| `MAX_UPLOADED_FILE_MB` | `50` | Maximum size per PDF |
| `CHROMA_DB_DIR` | `storage/chroma` | Where ChromaDB stores its data |
| `CHROMA_COLLECTION_NAME` | `study_notes` | ChromaDB collection name |
| `DEMO_MODE` | `false` | Per-session collections and a question cap for public demos |
| `DEMO_MAX_QUESTIONS` | `20` | Questions per session in demo mode |

For notes in Finnish or other languages, set `EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2`, lower `CHUNK_SIZE` to around 500, and process the PDFs again. The distance of every retrieved chunk is shown in the UI, which helps when tuning the threshold for your own notes.

## Testing

```bash
pytest                                  # 119 tests, no network access needed
pytest --cov --cov-report=term-missing  # with coverage
ruff check . && ruff format --check .   # lint and format
python scripts/evaluate_retrieval.py    # retrieval evaluation, downloads the models
```

The tests use small in-memory PDFs, a temporary ChromaDB and fake embedding and chat models. They cover PDF validation, chunking, embeddings, storage and retrieval, prompts, citations, configuration, the evaluation metrics, the full pipeline, and the Streamlit app through `AppTest`.

## Deployment

The app deploys for free on Streamlit Community Cloud with a free Gemini or Groq key. Step-by-step instructions are in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Limitations and next steps

- **Scanned PDFs** have no selectable text and need OCR, which is not included yet.
- **Slides and tables.** Slide decks produce short chunks, repeated headers and footers add noise, and tables, formulas and images are not interpreted.
- **Retrieval is vector only.** Hybrid keyword and vector search, or a reranker, would help with exact terms such as course codes.
- **The evaluation is small.** Thirty questions on one document, written by the same author as the notes, so the numbers are optimistic. Answer quality, as opposed to retrieval, is not measured yet.
- **Privacy.** Retrieved note text is sent to the configured chat provider. Use Ollama to keep everything on your machine.
- **Single user locally.** There are no accounts, and outside demo mode all notes share one collection.

Ideas for next steps: OCR for scanned notes, separate collections per course, deleting single documents, chat history, and generated flashcards or quizzes.

## License

This project is intended for educational and portfolio use.
