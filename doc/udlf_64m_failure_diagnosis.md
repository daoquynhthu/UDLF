# UDLF 64M Failure Diagnosis

## Decision

The current UDLF configuration is not competitive with the matched-scale
Mamba baseline on FineWeb-Edu. This is not explained by validation noise,
diffusion sampling noise, a nonfunctional persistent state, or an injection
shortcut. The strongest diagnosed failure is representational collapse of the
latent slots, compounded by poor parameter allocation and a training protocol
that truncates credit assignment much more aggressively than the design
document specifies.

## Confirmed Evidence

The completed runs contain no NaN or Inf metrics. On the same first 128
validation sequences at length 512:

| model | loss | perplexity | batch-loss SE |
| --- | ---: | ---: | ---: |
| UDLF, ODE evaluation | 5.1396 | 170.65 | 0.0512 |
| Mamba | 4.3899 | 80.63 | 0.0689 |

The loss gap is 0.7497. Mamba also used fewer parameters and fewer training
tokens in the original 3000-step comparison, so those mismatches do not explain
UDLF's deficit.

## Diagnosed Failure Mechanisms

### 1. Latent-slot collapse

At the UDLF step-3000 checkpoint, the 16 final slots have:

- mean pairwise cosine similarity: 0.941;
- centered slot RMS: 0.211;
- participation rank: 1.82 out of 16.

The learned initial state has participation rank 7.74, but recurrent dynamics
collapse it to fewer than two effective directions. The model therefore pays
for 16 slots and latent self-attention while obtaining roughly two-dimensional
slot diversity. The implementation has no persistent slot identity or explicit
anti-collapse mechanism; self-attention and common token conditioning favor
homogenization.

### 2. Most parameters are vocabulary matrices

UDLF parameter allocation is:

| group | parameters |
| --- | ---: |
| input embedding | 25,731,584 |
| untied output matrix | 25,731,584 |
| injection | 1,574,912 |
| prior dynamics | 9,708,032 |
| readout | 5,302,865 |
| initial state | 8,192 |

Only 16.59M of 68.06M parameters remain after the two vocabulary matrices.
Mamba ties its vocabulary matrix and has about 31.5M non-embedding parameters.
The nominal 64M comparison therefore gives Mamba roughly twice the trainable
core capacity.

### 3. Credit assignment is truncated at 64 tokens

Formal UDLF training detaches the carried state every 64 tokens. Mamba receives
full 512-token backpropagation. The architecture document instead specifies a
wide random truncation range of 64-1024. The checkpoint does use long-term
state, but training cannot assign credit across those boundaries:

- correct carry loss: 5.2006;
- reset state every 64 tokens: 5.3338;
- shuffle carried state across samples every 64 tokens: 5.4056.

The state contains useful document-specific information, making truncated
credit assignment a real limitation rather than a harmless memory optimization.

### 4. The formal configuration departed from the intended capacity regime

The design document recommends an 8k-16k vocabulary, random truncation lengths
from 64 to 1024, and K=4 as the default integration depth. The formal run used
vocabulary 50,257, fixed truncation 64, K=2, and an untied output matrix. Each
change had an engineering motivation, but together they produced a materially
different and weaker capacity allocation than the architecture document.

## Ruled-Out Explanations

### Diffusion noise is not the main loss source

Across eight paired evaluation seeds, stochastic loss is 5.2012 with SE
0.00018. ODE evaluation is 5.2006, an improvement of only 0.0006. Removing
noise at inference cannot close the gap.

### The persistent state is not unused

Resetting or shuffling state degrades loss, and a per-token stateless model has
loss 6.3730. The persistent state contributes materially.

### Injection is not a dominant shortcut

The frozen checkpoint with injection but no prior dynamics has loss 10.0095.
Prior dynamics without injection has loss 6.3489. Both components are required;
the model has not bypassed dynamics through direct observation injection.

## Remediation Order

1. Add explicit learned slot identities at initialization and at each dynamics
   step; add slot-diversity diagnostics as a hard training gate.
2. Tie input/output embeddings or use a 16k tokenizer so that at least 30M
   parameters are available to the UDLF core at the same total size.
3. Replace fixed 64-token detachment with randomized 64-512 truncation and
   periodic full-512 gradient steps or recurrent checkpointing.
4. Restore K=4 only after the first three changes; K=2 is not the primary
   diagnosed failure and restoring it alone would mainly reduce throughput.
5. Run a staged ablation: slot identity, then parameter reallocation, then
   credit horizon. Each stage must report fixed-sample validation loss, slot
   participation rank, carry/reset/shuffle deltas, throughput, and memory.

The next full 3000-step run should not begin until a medium-scale gate shows
slot participation rank materially above the current 1.82 and validation loss
improves beyond the existing checkpoint at matched token count.
