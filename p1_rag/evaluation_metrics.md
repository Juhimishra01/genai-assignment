# Evaluating Retrieval and Answers

Retrieval quality and answer quality must be measured separately, because a good answer can hide poor retrieval and the reverse.

## Retrieval metrics

Hit Rate records whether at least one relevant chunk appears in the top-k results. Recall@k measures the fraction of all relevant chunks that were retrieved. Mean Reciprocal Rank, or MRR, averages the reciprocal of the rank of the first relevant result, so a relevant chunk at rank 2 scores 0.5. Normalised Discounted Cumulative Gain, or nDCG, rewards placing relevant chunks near the top by dividing each gain by the logarithm of its rank.

## Answer metrics

Faithfulness, also called groundedness, measures whether every claim in the answer is supported by the retrieved context. Answer relevance measures whether the answer actually addresses the question. Both are commonly scored by a language model acting as a judge. Exact Match and F1 compare the answer with a gold answer when one exists.

## Judge biases

A language model judge has known biases. Position bias means it may prefer whichever answer is shown first, and it is checked by running each pair in both orders. Verbosity bias means it may prefer longer answers. Self-enhancement bias means it may prefer outputs from its own model family, so the judge should come from a different family than the generator. Cohen's kappa is used to measure how well the judge agrees with human labels beyond chance.
