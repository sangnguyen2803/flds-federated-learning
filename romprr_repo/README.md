# Information

This folder contains the configurations, logs and notebook of the visualisations of lab 1 to 2 (up to the experiments by modifying parameters).  

Since i just drag and drop my folders from my local (other) repository there will only be one commit, as it is only the "clean" result of lab 1 and 2.

## Content

`/runs` contains the logs of the **CsvLogger** from fluke. The base runs from lab 1 and the baseline run from lab 2 are all in this folder.  

`/runs/<subfolder>` are subfolders specific to the parameter tuning experiments, for example for the clients parameter the path to the experiments will be `/runs/clients/X` where ``X`` is equal to the number of clients.

This architecture is als used for the config folder which contains the .yaml configuration files for the experiments.

The notebooks in `/notebooks` are made to visualise the results of the experiments metrics wise.

The `/models` and `/dataset` contains the python files used for custom models and datasets.

---

## Vertical Federated Learning (VFL) — Training Flow

In VFL (Split Learning), the model is split in two: each client trains a **BottomModel** on its local feature subset, and the server trains a **TopModel** on the concatenated embeddings. Here is the full flow for **one mini-batch** during training:

### Step 1 — Client → Server (forward: embeddings)

Each client holds a different slice of columns for the same rows. They all process the **same batch index** in sync.

```
Client 0: X[:, [0,1,2,3]]  →  BottomModel_0  →  emb_0  (shape: [64, 32])
Client 1: X[:, [4,5,6,7]]  →  BottomModel_1  →  emb_1  (shape: [64, 32])
Client 2: X[:, [8,9,10]]   →  BottomModel_2  →  emb_2  (shape: [64, 32])
```

Each client caches its embedding (`_cached_embedding = emb`) because it needs it later for backprop. Then a **detached copy** (`emb.detach().requires_grad_(True)`) is sent to the server. The detach breaks the computation graph — the server's graph and the client's graph are **separate** (this simulates the network boundary).

### Step 2 — Server processes (forward + backward)

The server concatenates all embeddings and runs its TopModel:

```
concat = [emb_0 | emb_1 | emb_2]    →  shape: [64, 96]
logits = TopModel(concat)            →  shape: [64, 2]
loss = CrossEntropyLoss(logits, y)   →  scalar
loss.backward()                      →  computes concat.grad  (shape: [64, 96])
```

Because `concat.retain_grad()` is called, PyTorch stores the gradient even though `concat` isn't a leaf tensor. The server then **slices** this gradient back apart:

```
concat.grad[:, 0:32]   →  grad_0  (for Client 0)
concat.grad[:, 32:64]  →  grad_1  (for Client 1)
concat.grad[:, 64:96]  →  grad_2  (for Client 2)
```

The server updates its own TopModel weights with `optimizer.step()`.

### Step 3 — Server → Client (backward: gradients)

Each gradient slice is sent back to the corresponding client. The client then does:

```python
optimizer.zero_grad()
_cached_embedding.backward(grad_embedding)   # backprop through BottomModel
clip_grad_norm_(max_norm=1.0)                # prevent explosion
optimizer.step()                             # update BottomModel weights
```

The key: `_cached_embedding` still has the **original** computation graph attached (input → fc1 → relu → fc2 → relu → embedding). Calling `.backward(grad)` on it propagates the server's gradient through the BottomModel layers, computing gradients for `fc1` and `fc2` weights.

### Summary of what crosses the boundary

| Direction | What is sent | Shape | Raw data exposed? |
|---|---|---|---|
| Client → Server | Embedding (activations) | `[batch, 32]` | ❌ No raw features |
| Server → Client | Gradient of loss w.r.t. embedding | `[batch, 32]` | ❌ No labels |

No party ever sees the other's private data — only intermediate activations and gradients are exchanged.