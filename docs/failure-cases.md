# ContextForge Failure Cases

These are documented cases where ContextForge does not help or may produce worse results.
Knowing the failure modes is part of using the library correctly.

## 1. Very small token budgets (< 200 tokens)
When the budget is extremely tight, the allocator gives each source so few tokens
that even relevant sentences get dropped. The context becomes too sparse to answer.
**Mitigation:** Set top_n=1 and give all budget to the highest-scored source.

## 2. Ambiguous or very short queries
A query like "tell me more" or "explain" gives the SemanticScorer no signal.
All sources score similarly and reranking is essentially random.
**Mitigation:** Require minimum query length (>5 words) before calling engine.build().

## 3. Multi-hop questions where evidence is split across low-scoring chunks
HotpotQA multi-hop questions sometimes need two chunks that individually score low
but together answer the question. The reranker may drop one.
**Mitigation:** Increase top_n. Don't compress multi-hop pipelines aggressively.

## 4. Code and structured data with tight budgets
ContextForge never compresses code or JSON. If the only relevant source is a 5k-token
code file and the budget is 1k, the file is included verbatim and budget is exceeded.
**Mitigation:** Handle code sources separately or increase budget.

## 5. Approximate token counting mismatch
tiktoken cl100k_base overestimates Anthropic tokens by ~5% in some cases.
A context that appears to be 7,900 tokens may actually be 7,500 or 8,200.
**Mitigation:** Add a 10% safety margin to your budget. budget=8000 effectively gives you ~7,200 safe tokens.

## 6. Repeated similar documents
If all retrieved sources say the same thing (e.g., scraped from similar pages),
the reranker will rank them all highly and context will be redundant.
**Mitigation:** Deduplicate sources before passing to engine.build().

## 7. Very long single documents (> 10k tokens)
Extractive compression on a 10k-token document with a 1k budget may select
sentences that are individually relevant but lack surrounding context.
**Mitigation:** Pre-chunk long documents before using ContextForge.
