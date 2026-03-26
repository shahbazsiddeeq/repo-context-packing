## Pilot Result 1: RepoEval retrieval

Subset: first 5 line-level examples from RepoEval
Retriever: BM25
Baseline: fixed-size chunking
Method: structure-aware chunking

Fixed:
- MRR = 0.8000
- Recall@1 = 0.6000
- Recall@5 = 1.0000

Structure-aware (140 tokens):
- MRR = 0.9000
- Recall@1 = 0.8000
- Recall@5 = 1.0000

Observation:
- Structure-aware chunking improves retrieval.
- Very small structure chunks (60, 100) hurt badly.
- Current packing methods do not outperform top-k on this subset.

## Pilot Result 2: RepoEval retrieval on 20 examples

Fixed-size chunking:
- MRR = 0.6014
- Recall@1 = 0.5000
- Recall@5 = 0.7000

Structure-aware chunking (140 tokens):
- MRR = 0.5333
- Recall@1 = 0.4000
- Recall@5 = 0.7000

Observation:
- The 5-example gain for structure-aware chunking did not hold on the 20-example subset.
- Fixed-size chunking is currently the stronger retrieval baseline.
- RepoEval line-level tasks appear to benefit from local contiguous context.
- Current packing methods do not improve over top-k on this subset.
- Next step: test a hybrid chunking strategy combining fixed local chunks and structure-aware chunks.

## Pilot Result 4: RepoEval locality reranking on 50 examples

Fixed BM25 baseline:
- MRR = 0.5501
- Recall@1 = 0.4400
- Recall@5 = 0.6800

Best locality reranking:
- same_file_bonus = 100.0
- line_distance_weight = 0.2
- MRR = 0.6320
- Recall@1 = 0.5200
- Recall@5 = 0.7400

Observation:
- Locality-aware reranking is the strongest method tested so far on the 50-example subset.
- Fixed chunking remains the best retrieval substrate.
- Current structure-aware chunking and packing variants do not outperform the fixed+locality baseline.
- The gain appears substantial enough to justify scaling to a larger subset.

## Pilot Result 5: RepoEval locality reranking on 100 examples

Fixed BM25 baseline:
- MRR = 0.5865
- Recall@1 = 0.4900
- Recall@5 = 0.6900

Best locality reranking:
- same_file_bonus = 150.0
- line_distance_weight = 0.2
- MRR = 0.6565
- Recall@1 = 0.5600
- Recall@5 = 0.7600

Observation:
- Locality-aware reranking remains the strongest method tested so far on the 100-example subset.
- The gain persists at larger scale, so it is unlikely to be only a small-sample effect.
- Fixed chunking is the best retrieval substrate among the tested chunking strategies.
- Structure-aware, hybrid, and packing-only variants do not outperform fixed+locality on this RepoEval line-level setup.