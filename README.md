# HTSIR: Improving Long-Context Summarization with Multi-Granularity Retrieval Optimization

Official implementation of **HTSIR** from our AAAI 2026 paper:

> **Improving Long-Context Summarization with Multi-Granularity Retrieval Optimization**  
> Xueyu Chen, Kaitao Song, Zifan Song, Dongsheng Li, Cairong Zhao  
> *Proceedings of the Fortieth AAAI Conference on Artificial Intelligence (AAAI-26)*

## Overview

**HTSIR** is a hierarchical multi-granularity retrieval-augmented generation framework designed for long-context question answering and summarization. It consists of four main stages:

1. **HTSIR Tree Construction**: Organizes documents into multi-granularity representations—fine-grained leaf chunks ($x_{ji}$, length $l$), intermediate summaries ($p_k$, grouped over $r$ consecutive chunks), and coarse-grained section-level root summaries ($\tilde{s}_j$)—supporting both structured documents (e.g., academic papers in QASPER) and unstructured long texts.
2. **Collapsed Tree Search**: Aggregates nodes across all granularity levels into a unified search pool and retrieves the Top-$k$ most relevant nodes via dense semantic similarity.
3. **Two-Stage Reranking**: Refines candidate nodes using a query-aware reranker (e.g., `BAAI/bge-reranker-large` locally or Cohere Rerank) to prioritize the most relevant evidence.
4. **Response Generation**: Feeds the top reranked multi-granularity contexts into an LLM (e.g., GPT-4o-mini or open-source LLMs) to generate accurate answers.

## Installation

Python 3.9+ is recommended.

```bash
git clone https://github.com/chenxueyu23a/HTSIR.git
cd HTSIR
pip install -r requirements.txt
```

Set your API key via environment variables (never hardcode keys in source files):

```bash
# Linux / macOS
export OPENAI_API_KEY="your-openai-api-key"
# Optional: custom OpenAI-compatible endpoint
# export OPENAI_BASE_URL="https://your-endpoint/v1"
# Optional: only required if using Cohere reranker
# export COHERE_API_KEY="your-cohere-api-key"
```

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY="your-openai-api-key"
```

## Quick Start

### 1. Plain-Text Documents

```python
from htsir import RetrievalAugmentation, RetrievalAugmentationConfig

config = RetrievalAugmentationConfig(
    tb_max_tokens=100,
    use_reranker=True,
    reranker_type="local",
    reranker_model_name="BAAI/bge-reranker-large",
    reranker_top_k=4,
)
ra = RetrievalAugmentation(config=config)

with open("demo/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

ra.add_documents(text, chunk_group_size=2)

question = "How did Cinderella reach her happy ending?"
context, answer = ra.answer_question(question=question, top_k=10)
print("Answer:", answer)
```

### 2. Saving and Loading a Tree

```python
SAVE_PATH = "outputs/cinderella.pkl"
ra.save(SAVE_PATH)

ra = RetrievalAugmentation(config=config, tree=SAVE_PATH)
context, answer = ra.answer_question(question=question, top_k=10)
```

### 3. Structured Documents (QASPER Evaluation)

Download the QASPER dataset (e.g., `qasper-dev-v0.3.json` or `qasper-train-v0.2.json`) and run:

```bash
python eval_qasper.py \
    --data_path data/qasper-dev-v0.3.json \
    --output_path outputs/qasper_results.json \
    --chunk_size 100 \
    --chunk_group_size 2 \
    --top_k 10 \
    --reranker_top_k 4
```

You can also disable reranking with `--no_reranker` or evaluate a slice of papers with `--start_idx` and `--end_idx`.

## Extending HTSIR with Custom Models

HTSIR allows plugging in custom summarization, QA, and embedding models by subclassing `BaseSummarizationModel`, `BaseQAModel`, and `BaseEmbeddingModel`:

```python
from htsir import (
    BaseEmbeddingModel,
    BaseQAModel,
    BaseSummarizationModel,
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
)


class CustomSummarizationModel(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        return "Your summary here"


class CustomQAModel(BaseQAModel):
    def answer_question(self, context, question):
        return "Your answer here"


class CustomEmbeddingModel(BaseEmbeddingModel):
    def create_embedding(self, text):
        return [0.0] * 768


config = RetrievalAugmentationConfig(
    summarization_model=CustomSummarizationModel(),
    qa_model=CustomQAModel(),
    embedding_model=CustomEmbeddingModel(),
)
ra = RetrievalAugmentation(config=config)
```

See [demo.ipynb](demo.ipynb) for an interactive walkthrough.

## Project Structure

```text
HTSIR/
├── htsir/
│   ├── __init__.py
│   ├── EmbeddingModels.py        # SBERT & OpenAI embedding wrappers
│   ├── SummarizationModels.py    # LLM summarization wrappers
│   ├── QAModels.py               # LLM & UnifiedQA answer generation wrappers
│   ├── Retrievers.py             # Base retriever interface
│   ├── FaissRetriever.py         # Flat FAISS baseline retriever
│   ├── tree_structures.py        # Node and Tree data structures
│   ├── tree_builder.py           # Multi-granularity hierarchical tree construction
│   ├── cluster_utils.py          # UMAP + GMM clustering (HTSIR_Clustering)
│   ├── cluster_tree_builder.py   # Cluster-based tree builder
│   ├── tree_retriever.py         # Collapsed & layer-wise tree retrieval
│   ├── RetrievalAugmentation.py  # End-to-end HTSIR pipeline & rerankers
│   └── utils.py                  # Tokenization, chunking, and distance utilities
├── demo/
│   ├── sample.txt
│   └── arxivsample1.txt
├── eval_qasper.py                # QASPER benchmark evaluation script
├── demo.ipynb                    # Quickstart notebook
├── requirements.txt
├── LICENSE
└── README.md
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you find HTSIR useful in your research, please cite our AAAI 2026 paper:

```bibtex
@inproceedings{chen2026improving,
  title={Improving long-context summarization with multi-granularity retrieval optimization},
  author={Chen, Xueyu and Song, Kaitao and Song, Zifan and Li, Dongsheng and Zhao, Cairong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={36},
  pages={30315--30323},
  year={2026}
}
```

