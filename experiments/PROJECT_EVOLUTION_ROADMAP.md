# Project Evolution Roadmap

This is a simple history of how the chess-transformer project evolved, what changed at each stage, and which ideas turned out to matter most.

## 1. First phase: prove a model can play legal chess at all

Early work started with text-style and backbone-heavy approaches. The important breakthrough from this phase was not strength, but proof that a model could map board states to legal moves consistently.

Simple example:
- Before: treat chess more like a generic sequence problem.
- After: use a chess-specific board encoder and legal move masking.
- Why it mattered: the model stopped wasting capacity on impossible moves and started behaving like a real chess policy model.

Main finding:
- Legal move masking and chess-native inputs were foundational. Without them, everything downstream is weaker and noisier.

## 2. Architecture breakthrough: the spatial policy head

One of the earliest major breakthroughs was replacing a flat move classifier with a spatial policy head.

Simple example:
- Flat head: score all moves as unrelated labels.
- Spatial head: think in terms of "from square", "to square", and promotion, which matches how chess moves actually work.

Why this was revolutionary:
- It gave a huge jump in move prediction quality with fewer wasted parameters.
- The README shows this as one of the biggest early step changes:
  - flat head around 11% top-1
  - spatial head around 30% top-1 on the same small-scale setup

Main finding:
- The model got much stronger when the output structure matched chess geometry.

## 3. Scaling the native transformer

After the spatial head worked, the project scaled up the native transformer itself.

Simple example:
- Small model -> medium model
- Result: medium outperformed small by a meaningful margin instead of saturating immediately

What this changed:
- It showed the project was no longer just a toy architecture demo.
- More capacity could be converted into more chess skill, provided the data and training setup were good enough.

Main finding:
- Model size helped, but only when paired with enough data and the right training recipe.

## 4. Data quality beat clever loss functions

An important mid-stage lesson was that better positions mattered more than fancy objectives.

Simple example:
- Comparing policy CE vs action-value style targets on weak/random positions did not help much.
- Switching to stronger real-game or stronger-engine data gave much larger gains.

Why this was helpful:
- It prevented wasted time on over-optimizing the loss while feeding mediocre data.

Main finding:
- Better supervision source > more exotic objective, at least early on.
- Position quality and diversity were major bottlenecks.

## 5. Search improved play, but exposed the value bottleneck

Search experiments showed that raw policy alone was not enough. Value-guided reranking improved gameplay, but deeper search often underperformed because the value head was not yet trustworthy enough.

Simple example:
- Policy argmax: weaker play
- Value reranking over top-k moves: clearly better
- Deeper minimax/MCTS with noisy value: can make things worse instead of better

Why this was a breakthrough:
- It clarified that policy quality alone does not determine engine strength.
- It also revealed the next bottleneck very clearly: value estimation quality.

Main finding:
- Search helps only if the value head is good enough.
- A weak value head can poison deeper search.

## 6. Bigger data pipelines unlocked the next stage

Later work moved from relatively small datasets to massive Stockfish-labeled shard pipelines and large Lichess/engine-derived corpora.

Simple example:
- Early training: tens or hundreds of thousands of positions
- Later training: millions to 10M+ positions, with shard-based loaders and larger offline corpora

Why this mattered:
- It turned the project from a small supervised prototype into a real scaling effort.
- It also made later architectural wins meaningful, because the model finally had enough examples to learn from.

Main finding:
- Data scale and data system engineering became first-class research contributions, not just support work.

## 7. Strong modern recipe emerges: compact vocab, board flip, distributional value, balanced sampling

Around the 150s-167 range, the project consolidated many improvements into a more serious training recipe:
- compact move vocabulary
- board-flip / side-to-move normalization
- distributional value head
- phase-balanced sampling
- auxiliary losses
- larger 200M-class models

Simple example:
- Instead of asking the model to learn separate white-side and black-side patterns, board flip makes many positions look like the same problem from side-to-move perspective.

Why this was novel/helpful:
- These are the kinds of changes that usually do not look flashy individually, but together they create a much cleaner optimization problem.

Main finding:
- The project matured from "try one neat trick" into "assemble a coherent training system."

## 8. Distributional value was one of the most novel ideas

`exp168` is one of the most conceptually interesting experiments: take a strong existing model and surgically replace the simple 3-class WDL value head with a richer 128-bin distributional head.

Simple example:
- Old value head: both +0.5 and +9.0 can look like "winning".
- Distributional value: distinguishes slightly better from overwhelmingly better.

Why this is revolutionary:
- It targets the exact place where search quality usually collapses.
- It is a direct attempt to convert a "coarse chess instinct" into a more precise search signal.

Main finding:
- Value quality is likely one of the highest-leverage levers for Elo once policy is decent.

## 9. Late-stage ablation breakthrough: SwiGLU + chess relative bias

The recent sequence `exp169 -> exp170` produced a very important architectural conclusion.

What was tested:
- Micro-scale rapid ablations in `exp169`
- Then confirmation at medium scale in `exp170`

Simple example:
- Baseline transformer: generic GELU + absolute position embeddings
- Improved transformer: SwiGLU feedforward + chess-aware relative geometry bias

Why this was a breakthrough:
- The gain showed up first in the tiny fast experiments, then held up when scaled.
- That is exactly what you want from an ablation program: cheap screening first, expensive confirmation second.

Main finding:
- SwiGLU + relative bias appears to be the dominant architecture improvement from the recent ablation cycle.

## 10. Data scaling was real, not theoretical

`exp171` tested whether simply exposing the model to more shards and more steps would help.

Simple example:
- Train on 2 shards -> okay result
- Train on 4 to 8 shards -> higher top-1 and less obvious overfitting

Why this mattered:
- It confirmed that the earlier plateau was not purely architectural.
- The model was starving for diversity.

Main finding:
- More diverse training data broke through earlier ceilings.

## 11. Cosine learning rate was a major practical breakthrough

`exp172` may be one of the most practically important experiments so far.

Simple example:
- Constant LR: accuracy peaks and then drifts down late in training
- Cosine LR with warmup: accuracy finishes stronger instead of getting noisier at the end

Why this was such a strong result:
- It improved final performance without needing a brand-new model or brand-new data source.
- It directly addressed a real observed failure mode: late-stage instability.

Main finding:
- Better scheduling gave a large win for cheap engineering effort.
- This is one of the highest ROI training improvements in the recent history.

## 12. Scale-up was validated at 50M

`exp173` is one of the clearest recent breakthroughs.

What happened:
- The recent best recipe was scaled from about 25.9M params to about 50M params.
- The 50M model beat the smaller baseline even on a harder held-out shard.

Simple example:
- 25.9M baseline: about 17.48% top-1 in the reference note
- 50M model: 18.36% top-1 on a harder held-out evaluation shard

Why this is a breakthrough:
- It is strong evidence that the project is still capacity-limited.
- Bigger models are not just matching smaller ones; they are continuing to buy real accuracy.

Main finding:
- The 50M scale-up is validated.
- The model likely has not plateaued yet.

## 13. The next idea: longer training, not just bigger models

`exp174` is the natural follow-up to `exp173`.

Core idea:
- At 10K steps, the 50M model had only seen a small fraction of the available 8M-position pool.
- So the next bet is: keep the same successful architecture and train much longer before assuming you need another major redesign.

Simple example:
- 10K-step run used only about 8% of the data pool
- 30K-step run would expose the model to about 24%

Why this is helpful:
- It reframes the problem from "we need another trick" to "we may simply be undertraining the system we already have."

Main finding:
- The current frontier is not just architecture search. It is training utilization.

## Biggest breakthroughs so far

If I had to rank the most important breakthroughs in plain English:

1. Spatial policy head
- The first huge architecture jump.
- It changed the model from a generic classifier into something more chess-native.

2. Data quality and scale
- Better and larger datasets repeatedly mattered more than many smaller modeling tweaks.

3. Search exposed the value bottleneck
- This changed the project’s understanding of where Elo was being lost.

4. Distributional value modeling
- One of the most novel ideas in the repo.
- Potentially very high leverage for search strength.

5. SwiGLU + chess relative bias
- Recent architecture winner that survived both cheap and larger-scale testing.

6. Cosine LR schedule
- One of the best engineering-return improvements: simple, cheap, and clearly effective.

7. 50M model scale-up
- Strong evidence the project can still benefit from more capacity.

## Simplest summary of how the project evolved

The project started by proving that a chess-native transformer could produce legal and meaningful moves at all. Then it found a much better output structure with the spatial policy head. After that, it learned that data quality, data scale, and value quality matter more than many clever isolated tricks. The recent stage has been about turning those lessons into a stronger recipe: better architecture, better schedule, bigger data, and larger models. The latest experiments suggest the project is still undertrained rather than fundamentally stuck.

## If you want the one-sentence thesis

The project evolved from "can a transformer play legal chess?" into "how do we systematically scale a chess-native policy/value/search system without wasting capacity on bad outputs, bad data, or bad optimization?"
