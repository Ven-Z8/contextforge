## Benchmark Gates

**Dataset:** Natural Questions dev split, 100 examples

| Gate | Status | Evidence |
|------|--------|----------|
| Vector recall preserved | PASS | drop=0.000; max=0.010 |
| Vector token reduction | PASS | reduction=40.3%; min=30.0% |
| ContextForge token budget | FAIL | violations=2; budget=4000 |
| Qdrant hybrid recall preserved | FAIL | drop=0.020; max=0.010 |
| Qdrant hybrid token reduction | PASS | reduction=56.6%; min=30.0% |

A failed gate blocks broad benchmark claims. It does not block publishing the result as an honest limitation.
