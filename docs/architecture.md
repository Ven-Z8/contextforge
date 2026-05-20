# ContextForge Architecture

## System Overview

```
Query + Sources
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ContextEngine                            │
│                                                                 │
│  Stage 1: SemanticScorer                                        │
│  ─────────────────────────────────────────────────────────      │
│  Bi-encoder (all-MiniLM-L6-v2)                                  │
│  Cosine similarity filter → keeps top_k candidates (default 20) │
│  Fast: ~5ms per query for 100 docs                              │
│                   │                                             │
│                   ▼                                             │
│  Stage 2: CrossEncoderReranker                                  │
│  ─────────────────────────────────────────────────────────      │
│  Cross-encoder (ms-marco-MiniLM-L-6-v2)                         │
│  Precision rerank → keeps top_n (default 5)                     │
│  Slower but more accurate than bi-encoder                       │
│                   │                                             │
│                   ▼                                             │
│  Stage 3: BudgetAllocator                                       │
│  ─────────────────────────────────────────────────────────      │
│  Distributes token_budget proportional to rerank scores         │
│  Higher-scored sources get more tokens                          │
│  Minimum 50 tokens guaranteed per source                        │
│                   │                                             │
│                   ▼                                             │
│  Stage 4: ContentTypeRouter + CompressionEngine                 │
│  ─────────────────────────────────────────────────────────      │
│  Router detects: PROSE | CODE | STRUCTURED                      │
│  CODE + STRUCTURED → verbatim (no compression ever)             │
│  PROSE → extractive sentence-level compression to budget        │
│                   │                                             │
│                   ▼                                             │
│         ContextWindow                                           │
│  ─────────────────────────────────────────────────────────      │
│  Rendered text with [Source: path | id=...] labels              │
│  Full attribution: score, compression_ratio, original_tokens    │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
  LLM (your_llm(window.render()))
```

## Key Design Decisions

### 1. Extractive only — no summarization
Every token in the output is verbatim from the original source.
Zero hallucination risk from the compression step itself.

### 2. Two-stage retrieval (bi-encoder → cross-encoder)
Bi-encoder is fast (O(n) dot product) — use it to filter 100→20.
Cross-encoder is accurate but slow (O(n) forward passes) — use it to rerank 20→5.
This is the standard pattern from BEIR / MS-MARCO research.

### 3. Content-type routing
Code and structured data (JSON, YAML, SQL) have precise syntax.
Sentence-level compression would corrupt them.
Hard rule: never lossy-compress code or structured data.

### 4. Injectable components
All pipeline stages accept custom implementations.
Swap all-MiniLM for voyage-3 embeddings, swap MiniLM cross-encoder for Cohere Rerank 4.
The engine orchestrates; the components are your choice.

### 5. Approximate token counting
tiktoken cl100k_base is not Anthropic's actual tokenizer.
Error margin: ~±5%. We document this and recommend a 10% safety margin on budgets.
Provider-native token counting (Anthropic beta) is a post-MVP addition.
