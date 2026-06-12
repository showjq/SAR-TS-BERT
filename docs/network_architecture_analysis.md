# Dual-Branch SAR-TS-BERT Network Architecture Analysis

## 1. Main Method Decision

This document treats `DualBranchSarBertEncoder` as the final paper method. The single-branch `SarBertEncoder` remains a valid code path, but in the paper figure it is now an ablation baseline rather than the primary architecture.

The architecture described here is grounded in the real `forward` paths in the repository.

## 2. Code Evidence

Core model files:

- `src/models/bert_sar_encoder.py`
  - `DualBranchSarBertEncoder`: final main encoder.
  - `CrossModalFusion`: bidirectional cross-attention fusion between scatter and physics branches.
  - `SarBertEncoder`: single-branch ablation baseline.
  - `positional_encoding`: sinusoidal temporal position encoding.
- `src/models/parcel_classifier.py`
  - `ParcelClassifier.forward`: downstream crop classification path.
- `src/models/monthly_classifier.py`
  - `MonthlyParcelClassifier.forward`: monthly multi-head classification path.
- `src/models/dora.py`
  - `DoRAAdapter`, `DualBranchDoRAEncoderLayer`: optional residual low-rank adapter path.

Training and inference files:

- `src/train/pretrain_encoder.py`: self-supervised masked reconstruction pre-training.
- `src/train/finetune_classifier_processed.py`: standard crop classification fine-tuning.
- `src/train/finetune_monthly_classifier.py`: monthly crop classification fine-tuning.
- `src/train/test_classifier.py`: classification evaluation.
- `src/infer/parcel_infer.py`: pixel and parcel inference, optional shapefile export.

Data files:

- `src/data/zarr_dataset.py`: `TemporalPixelDataset` for unlabeled pixel-level pre-training samples.
- `src/data/processed_dataset.py`: `ProcessedParcelDataset` for processed zarr classification samples.
- `src/data/monthly_dataset.py`: `MonthlyParcelDataset` for monthly labels.

Reference hyperparameters from existing experiment configurations:

- `d_model=256`
- `nhead=8`
- `layers=6`
- `ff=512`
- `dora_rank=32` when DoRA is enabled

## 3. Inputs

### 3.1 Pre-Training Input

Source: `TemporalPixelDataset` in `src/data/zarr_dataset.py`.

The raw pre-training zarr store is expected to contain:

- `observations`: `(T, H, W, C)`
- `timestamps`

For each sampled pixel, the dataset returns:

- `clean`: `(T, C)`
- `noisy`: `(T, C)`
- `time_idx`: `(T,)`
- `mask`: `(T,)`

After batching:

- `clean`: `B x T x C`
- `noisy`: `B x T x C`
- `time_idx`: `B x T`
- `mask`: `B x T`

For the final paper method, the expected semantic input is six-band SAR:

`VV, VH, VV/VH, H, alpha, A`

so `C=6`.

### 3.2 Fine-Tuning Input

Source: `ProcessedParcelDataset` in `src/data/processed_dataset.py`.

The processed zarr store contains:

- `time_series`: `(N, T, C)`
- `labels`: `(N,)`
- optional `parcel_ids`
- optional `class_names`

Each batch contains:

- `series`: `B x T x C`
- `time_idx`: `B x T`
- `label`: `B`

For the final dual-branch paper method, `C=6` is required because the encoder explicitly splits the first three and last three channels.

## 4. Final Main Encoder: DualBranchSarBertEncoder

Class: `DualBranchSarBertEncoder` in `src/models/bert_sar_encoder.py`.

The encoder uses a physically motivated two-stream design:

- Scatter branch: backscatter amplitude and ratio channels.
- Physics branch: polarimetric decomposition channels.
- Fusion: bidirectional cross-attention plus projection.

Let:

- `B`: batch size
- `T`: number of SAR timesteps
- `d`: model width, usually 256
- `C=6`: number of SAR bands

### 4.1 Band Split

In `DualBranchSarBertEncoder.forward`:

```python
scatter = noisy[..., 0:3]
physics = noisy[..., 3:6]
```

This gives:

- `scatter`: `B x T x 3`, bands `VV, VH, VV/VH`
- `physics`: `B x T x 3`, bands `H, alpha, A`

This split is the primary paper method design, because it separates radar return amplitude information from polarimetric scattering-mechanism information.

### 4.2 Scatter Branch

Code modules:

- `self.scatter_proj = nn.Linear(3, d_model // 4)`
- `self.scatter_encoder = nn.TransformerEncoder(..., num_layers=scatter_layers)`

Forward path:

1. Project scatter bands:
   - `Linear(3, d/4)`
   - `B x T x 3 -> B x T x d/4`
2. Compute sinusoidal PE:
   - `positional_encoding(time_idx, d/4)`
   - `B x T -> B x T x d/4`
3. Concatenate:
   - `B x T x d/4` + `B x T x d/4`
   - output `B x T x d/2`
4. Transformer branch encoder:
   - output `h_s = B x T x d/2`

With `d=256`, scatter branch width is `128`.

### 4.3 Physics Branch

Code modules:

- `self.physics_proj = nn.Linear(3, d_model // 4)`
- `self.physics_encoder = nn.TransformerEncoder(..., num_layers=physics_layers)`

Forward path:

1. Project polarimetric bands:
   - `Linear(3, d/4)`
   - `B x T x 3 -> B x T x d/4`
2. Compute sinusoidal PE:
   - `B x T x d/4`
3. Concatenate:
   - output `B x T x d/2`
4. Transformer branch encoder:
   - output `h_p = B x T x d/2`

With `d=256`, physics branch width is also `128`.

### 4.4 Bidirectional Cross-Modal Fusion

Class: `CrossModalFusion` in `src/models/bert_sar_encoder.py`.

Code:

```python
attn_s, _ = self.cross_attn_s2p(h_s, h_p, h_p)
attn_p, _ = self.cross_attn_p2s(h_p, h_s, h_s)
fused = torch.cat([h_s + h_p, attn_s, attn_p], dim=-1)
fused = self.fusion(fused)
return self.norm(fused)
```

The module computes two cross-attention directions:

- Scatter queries physics:
  - query `h_s`, key/value `h_p`
  - output `attn_s = B x T x d/2`
- Physics queries scatter:
  - query `h_p`, key/value `h_s`
  - output `attn_p = B x T x d/2`

Then it concatenates:

- `h_s + h_p`: shared aligned temporal representation, `B x T x d/2`
- `attn_s`: scatter-to-physics attended signal, `B x T x d/2`
- `attn_p`: physics-to-scatter attended signal, `B x T x d/2`

Concatenated tensor:

- `B x T x 3d/2`

Projection:

- `Linear(3d/2, d)`
- `LayerNorm(d)`

Output:

- `h_fused = B x T x d`

### 4.5 Fusion Encoder

The code computes:

```python
remaining_layers = num_layers - scatter_layers - physics_layers
```

If `remaining_layers > 0`, `h_fused` is further processed by:

- `self.fusion_encoder = nn.TransformerEncoder(..., num_layers=remaining_layers)`

Output remains:

- `B x T x d`

If `remaining_layers <= 0`, `fusion_encoder` is `None`, and the cross-attention fused output is used directly.

## 5. Pre-Training Path

Entry: `src/train/pretrain_encoder.py`.

### 5.1 Masking and Corruption

`TemporalPixelDataset` supports:

- random masking
- contiguous span masking
- physics-guided masking
- span + physics masking

For the final method figure, the key strategy is span + physics masking:

1. Compute temporal changes in VV and VH:
   - `diff_vv = |VV[t] - VV[t-1]|`
   - `diff_vh = |VH[t] - VH[t-1]|`
   - `change_rate = (diff_vv + diff_vh) / 2`
2. Convert high-change positions into higher span-start probabilities.
3. Mask contiguous time spans around physically informative change points.
4. Add Gaussian noise only to masked timesteps.

The code does not use a learned `[MASK]` token. It uses additive Gaussian corruption:

- `x_tilde[t] = x[t] + Normal(0, noise_scale * std)`

### 5.2 Reconstruction Head and Loss

In `DualBranchSarBertEncoder`:

- `self.head = nn.Linear(d_model, num_bands)`

Forward path:

- `h_fused`: `B x T x d`
- reconstruction head: `B x T x 6`
- masked token selection: `|Mask| x 6`

Training loss in `pretrain_encoder.py`:

```python
pred = model(noisy, time_idx, mask)
b = torch.where(mask)
target = clean[b[0], b[1]]
loss = F.mse_loss(pred, target)
```

Loss:

- `L_pre = MSE(pred_masked, clean_masked)`

Only masked timesteps contribute to pre-training loss.

## 6. Fine-Tuning Path

Entry: `src/train/finetune_classifier_processed.py`.

Classifier: `ParcelClassifier` in `src/models/parcel_classifier.py`.

When `use_dual_branch=True`, `ParcelClassifier.forward` manually runs the dual-branch components:

1. Split `series` into scatter and physics bands.
2. Apply branch projections and temporal positional encodings.
3. Run scatter and physics Transformer branches.
4. Run `cross_fusion`.
5. Optionally run `fusion_encoder`.
6. Mean-pool over time:
   - `B x T x d -> B x d`
7. Classification head:
   - `Dropout(0.2)`
   - `Linear(d, K)`
8. Output:
   - logits `B x K`

Loss:

- `F.cross_entropy(logits, y)`

The pre-trained encoder checkpoint is loaded with:

```python
model.encoder.load_state_dict(sd, strict=False)
```

## 7. Monthly Classification Variant

Class: `MonthlyParcelClassifier` in `src/models/monthly_classifier.py`.

This variant uses the same encoder representation but changes the supervised head:

1. Encode SAR sequence.
2. Mean-pool over time:
   - `B x T x d -> B x d`
3. For each month `m`, add a learned month embedding:
   - `pooled + month_embedding[m]`
4. Apply month-specific classifier:
   - `Linear(d, K)`
5. Stack logits:
   - `B x M x K`

Training loss:

- average cross-entropy over months.

This is a downstream task variant and should be shown only if the paper figure needs to emphasize monthly crop-state prediction.

## 8. Optional DoRA-Style Adapter

File: `src/models/dora.py`.

The implementation inserts residual low-rank adapters into Transformer layers:

- `Linear(d, rank)`
- `GELU`
- `Dropout`
- `Linear(rank, d)`
- residual addition

In `DualBranchDoRAEncoderLayer`, the adapter is applied after the attention block and after the FFN block. Tensor shape is unchanged:

- `B x T x d` or `B x T x d/2`, depending on the branch.

Important note: the implementation is adapter-like and does not explicitly decompose the original weight matrices in the forward pass. In the figure it is therefore shown as an optional training detail, not as the main architectural contribution.

## 9. Single-Branch Baseline

Class: `SarBertEncoder`.

This baseline maps all six channels jointly:

- `Linear(C, d/2)`
- concatenate positional encoding `d/2`
- Transformer Encoder x `N`
- reconstruction head or classification pooling

It is useful for ablation but is no longer the final paper main method.

## 10. Tensor Shape Summary

Let:

- `B`: batch size
- `T`: number of SAR timesteps
- `d`: model width
- `K`: number of crop classes
- `M`: number of months

Main dual-branch encoder:

| Stage | Operation | Shape |
| --- | --- | --- |
| Input | 6-band SAR sequence | `B x T x 6` |
| Split | Scatter / physics | `B x T x 3`, `B x T x 3` |
| Scatter projection | `Linear(3,d/4)` | `B x T x d/4` |
| Scatter token | concat with PE | `B x T x d/2` |
| Scatter encoder | Transformer x `Ns` | `B x T x d/2` |
| Physics projection | `Linear(3,d/4)` | `B x T x d/4` |
| Physics token | concat with PE | `B x T x d/2` |
| Physics encoder | Transformer x `Np` | `B x T x d/2` |
| Cross-attention | S->P and P->S | two `B x T x d/2` tensors |
| Fusion concat | `(S+P, S->P, P->S)` | `B x T x 3d/2` |
| Fusion projection | `Linear(3d/2,d)+LN` | `B x T x d` |
| Fusion encoder | Transformer x `Nf`, optional | `B x T x d` |
| Pre-train head | `Linear(d,6)` | `B x T x 6` |
| Masked select | `torch.where(mask)` | `|Mask| x 6` |
| Classification pool | temporal mean | `B x d` |
| Classifier | `Dropout+Linear(d,K)` | `B x K` |
| Monthly classifier | month embeddings + heads | `B x M x K` |

## 11. Key Designs to Highlight in the Paper Figure

The figure should emphasize:

1. **Physics-aware input factorization**: scatter bands and polarimetric decomposition bands are not treated as homogeneous channels.
2. **Two Transformer branches**: each branch learns temporal dynamics for a physically distinct SAR feature group.
3. **Bidirectional cross-attention fusion**: scatter attends to physics and physics attends to scatter.
4. **Masked SAR reconstruction pre-training**: high-change temporal spans are corrupted and reconstructed.
5. **Encoder transfer to crop classification**: the same fused temporal encoder is reused for supervised crop prediction.
6. **Single-branch and DoRA are secondary**: show them as optional/ablation details, not the main method.

## 12. Generated Diagram Files

- `docs/network_architecture_diagram.mmd`: Mermaid sketch.
- `docs/network_architecture_diagram.dot`: Graphviz source for paper rendering.
- `figures/network_architecture.svg`: rendered vector output if Graphviz is available.
- `figures/network_architecture.pdf`: rendered PDF output if Graphviz is available.

## 13. Rendering Note

Graphviz was installed in the existing `bert` conda environment and used to render the paper figure.

Graphviz version:

```text
dot - graphviz version 14.1.2 (0)
```

Rendered outputs:

- `figures/network_architecture.svg`
- `figures/network_architecture.pdf`
- `figures/network_architecture.png`

Rendering commands used:

```powershell
& "C:\Users\PC\.conda\envs\bert\Library\bin\dot.exe" -Tsvg docs\network_architecture_diagram.dot -o figures\network_architecture.svg
& "C:\Users\PC\.conda\envs\bert\Library\bin\dot.exe" -Tpdf docs\network_architecture_diagram.dot -o figures\network_architecture.pdf
& "C:\Users\PC\.conda\envs\bert\Library\bin\dot.exe" -Tpng docs\network_architecture_diagram.dot -o figures\network_architecture.png
```
