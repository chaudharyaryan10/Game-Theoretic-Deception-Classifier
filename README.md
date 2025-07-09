# Game-Theoretic-Deception-Classifier
GDetecting deceptive communication in the strategy game Diplomacy using graph neural networks, transformer-based attention, and game-theoretic modeling.
.

📌 Introduction
Diplomacy is a negotiation-heavy board game set in pre-WWI Europe, where players represent major powers and form or break alliances to win territorial control. The game is unique in that there are no dice or cards—only dialogue, making deception a key mechanic.

This project tackles deception detection: determining whether a player message is a lie. The task is challenging due to:

Only ~5% of messages being deceptive

Subtle and strategic nature of lies

High dependence on historical and social context

We use the QANTA Diplomacy Deception Dataset, annotated with ground-truth intent labels from the original sender, to train and evaluate two novel models.
We propose two complementary deep learning models that outperform prior baselines:

1. LieDetectorGAT (Graph Attention Network)
Models each Diplomacy game as a directed interaction graph

Players are nodes, messages are edges

Edges are enriched with:

BERT embeddings (768D)

Deception lexicon features (10D)

Game metadata (5D)

Uses GATv2Conv to propagate contextualized deception signals across the graph

2. HiS-Attention (Historical-Structured Attention)
A multimodal Transformer that fuses:

Current message (BERT)

Game state metadata

Last 10 dialogue messages

Uses query-based attention pooling to generate deception-aware fused representations

Each message includes:

Text content

Sender and receiver identity

Game metadata (season, year, score delta)

Optional dialogue history

Due to severe class imbalance (~5%), we apply class weighting and specialized architecture design.


Model Architectures
LieDetectorGAT
text
Copy
Edit
Game → Interaction Graph:
- Nodes: players (7 per game)
- Edges: messages (with BERT, deception lexicon, metadata)
- Graph Attention: GATv2 layers
- Classifier: 2-layer MLP on sender, receiver, edge embeddings

Training Setup:

Optimizer: AdamW, LR = 3e-4

Loss: BCEWithLogits + class weights

Batch size: 1 game

Library: PyTorch Geometric

HiS-Attention
text
Copy
Edit
Input Streams:
- Message: BERT encoding
- Game state: FFN over metadata
- History: BERT + Transformer over 10 past messages
Fusion:
- Query-based attention pooling
- Output passed to deception classifier

raining Setup:

Optimizer: Adam, LR = 1e-5

Weighted BCE Loss (pos class weight ≈ 21.2)

Batch size: 16

Scheduler + Gradient Clipping

📊Results
| Metric     | HiS-Attention | LieDetectorGAT |
| ---------- | ------------- | -------------- |
| Accuracy   | Higher        | Slightly Lower |
| Lie Recall | 22.0%         | **38.8%**      |
| Lie F1     | 22.0%         | **26.6%**      |
| Macro F1   | Competitive   | Competitive    |


