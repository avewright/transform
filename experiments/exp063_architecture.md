# `exp063_search_policy` Architecture

```mermaid
flowchart TD
    A[Board position<br/>python-chess Board] --> B[batch_boards_to_token_ids]
    B --> C[LearnedBoardEncoder<br/>board tokens]
    C --> D[input_proj<br/>encoder_dim to hidden_dim]

    D --> E[Prepend CLS token]
    E --> F[Add positional embeddings<br/>68 tokens total]
    F --> G[TransformerEncoder<br/>8 layers, 8 heads]
    G --> H[LayerNorm]

    H --> I[CLS hidden state]
    H --> J[Square hidden states<br/>tokens 4..67 = 64 board squares]

    subgraph PolicyHead[Spatial Policy Head]
        J --> K[from_proj]
        J --> L[to_proj]
        I --> M[global_proj]
        N[Move vocabulary<br/>all UCI moves] --> O[from_sqs / to_sqs / promo_types]
        O --> K
        O --> L
        O --> P[promo_embed]
        K --> Q[Gather from-square features<br/>for every move]
        L --> R[Gather to-square features<br/>for every move]
        M --> S[Broadcast global context]
        P --> T[Promotion features]
        Q --> U[Combine per-move features]
        R --> U
        S --> U
        T --> U
        U --> V[ReLU]
        V --> W[score_proj]
        W --> X[Policy logits<br/>one score per move in vocab]
    end

    subgraph ValueHead[Value Head]
        I --> Y[Linear]
        Y --> Z[ReLU]
        Z --> AA[Linear]
        AA --> AB[WDL logits<br/>win / draw / loss]
    end

    X --> AC[Apply legal move mask]
    AC --> AD[Softmax]
    AD --> AE[Move probabilities]
    AE --> AF[Top-k policy moves]

    AB --> AG[Softmax]
    AG --> AH[Value estimate]
```

## Notes

- The trunk is a single transformer over board tokens plus one `CLS` token.
- The policy head is a one-shot spatial scorer over the full move vocabulary.
- The value head reads only the final `CLS` representation.
- Search is not inside the network; inference uses masked policy probabilities and optional downstream reranking.
