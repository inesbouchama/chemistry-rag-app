# ChemRAG: Multimodal RAG for Advanced Chemistry Lectures

Retrieval-augmented question answering over slide-based organic chemistry lectures from the [Synthesis Workshop](https://www.synthesis-workshop.com/) series. Given a query, the system retrieves relevant lecture slides with a hybrid text/OCR retriever, links them to the matching YouTube timestamps, and lets a multimodal model answer questions grounded in the slide images and transcripts.

MSc semester project (EPFL–ETHZ), supervised by Prof. Philippe Schwaller. Full report: [`docs/semester_project_report.pdf`](docs/semester_project_report.pdf).

Live demo: https://chem-rag.streamlit.app

## Overview

- **Data**: 1,185 slide–transcript pairs from 174 lectures. Slides are extracted from the lecture videos and aligned with Whisper transcriptions.
- **Retrieval**: transcripts and EasyOCR slide text are embedded with `BAAI/bge-large-en-v1.5` and indexed in two FAISS indexes. Scores are combined as `S = (1 − w)·S_caption + w·S_OCR`, and same-speaker neighbouring slides are added to each hit.
- **Generation**: retrieved slides (image and caption) are passed to a multimodal model through OpenRouter.
- **Interface**: a Streamlit app for search, YouTube previews at the matched timestamp, and multi-turn chat about the retrieved slides.

## Repository structure

```
src/
├── rag_vf.py                   # ChemistryRAG: index building, hybrid retrieval, retrieval evaluation
└── streamlit_rag_chat_app.py   # Streamlit search and chat interface
docs/
└── semester_project_report.pdf
.devcontainer/                  # Codespaces / dev container configuration
requirements.txt
```

## Setup

Requires Python 3.11.

```bash
pip install -r requirements.txt
```

The slide metadata (`triplets_with_hf_paths.json`) and the pre-built indexes (`saved_indexes/`) are hosted in a private Hugging Face dataset (`ines-epfl-ethz/SW4retrieval`) and downloaded on first run. Access requires a Hugging Face token:

- for `rag_vf.py`, set the `HF_TOKEN` environment variable;
- for the Streamlit app, add `HF_TOKEN` to `.streamlit/secrets.toml`.

## Usage

### Web app

```bash
streamlit run src/streamlit_rag_chat_app.py
```

Run from the repository root: the indexes are extracted to `./saved_indexes`.

In the sidebar, select a provider and model, enter an API key, and click **Initialize Agent**. Search for a topic to retrieve slides and video clips, then ask follow-up questions in the chat panel. The number of retrieved results and the minimum spacing between timestamps can be set from the sidebar.

### Retrieval from Python

```python
from rag_vf import ChemistryRAG

rag = ChemistryRAG(save_dir="./saved_indexes", load_saved=True)
results, n_slides = rag.retrieve(
    query="What are examples of Buchwald-Hartwig coupling?",
    top_k=6,
    hybrid_weight=0.5,
    min_score_threshold=0.3,
    neighbor_window=1,
)
```

Each result contains the slide record (`caption`, `img_path`, `speaker`, `timestamp`, `youtube_url`), its hybrid score, and whether it is a main hit or a neighbour.

### Rebuilding the indexes

Set `load_saved=False` to recompute the OCR text, embeddings, and FAISS indexes from the slide images. This takes about 20 minutes and writes the results to `save_dir`.

## Results

Retrieval accuracy on the 54-question benchmark (true slide among retrieved slides):

| Method | Accuracy (%) | Avg. slides retrieved |
|---|---|---|
| SentenceTransformer + CLIP | 68.5 | 14.6 |
| BGE + CLIP | 71.2 – 77.8 | 12.1 – 15.0 |
| BGE + Tesseract OCR | 74.1 – 77.8 | 6.6 – 7.2 |
| **BGE + EasyOCR** (used here) | 83.3 – 96.3 | 10.3 – 24.0 |
| BGE + DONUT | 74.1 | 28.0 |

MCQ answer accuracy (%) on the hard benchmark (54 questions):

| Model | No context | True slide | RAG (4 slides) |
|---|---|---|---|
| Qwen2.5-VL | 33.3 | 42.6 | 41.6 |
| Gemini Flash 1.5 | 35.2 | 43.5 | 35.2 |
| Gemini Pro 1.5 | 29.6 | 49.1 | 44.4 |

See the report for the synthetic benchmark, prompting strategies, and variance across runs.

## Citation

> I. Bouchama. *Multimodal RAG for Advanced Chemistry: Targeted Information Retrieval and Question Answering from Slide-Based Lectures.* MSc semester project, EPFL–ETHZ, 2025.
