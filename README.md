# Pneumonia Attention Classifier

An intentionally small, reproducible PyTorch project for binary classification of
chest X-rays (`NORMAL` vs `PNEUMONIA`) using the Kaggle
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
dataset. It is designed as a portfolio-quality reference implementation: the
model is easy to read, the training/evaluation path is explicit, and every
prediction includes a learned **spatial attention map** that can be visualized.

> **Status:** This repository contains ready-to-run code and tests. No model was
> trained for this project checkout because the Kaggle data and a fully
> provisioned environment are not assumed to be available. Any reported metrics
> should be generated locally with the commands below, not inferred from this
> README.

## Highlights

- Compact CNN backbone with four convolution/pooling blocks.
- Actual spatial attention, not a post-hoc saliency claim: channel-wise average
  and maximum projections are convolved into a learned map `A ∈ [0, 1]`.
- Reproducible seed, class-weighted cross-entropy, AdamW, validation checkpointing.
- Held-out test metrics: accuracy, precision, sensitivity/recall, specificity,
  F1, and ROC-AUC (when both classes are present).
- `explain.py` produces input, attention, and overlay panels.
- Unit tests cover model output shapes, attention ranges, non-square features,
  and dataset image loading.

## Setup

```bash
cd pneumonia-attention-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

The code uses torchvision only as an optional ecosystem dependency; its simple
PIL transform keeps the dataset and tests readable and portable.

## Dataset

1. Create a Kaggle account and accept the dataset's terms.
2. Download `chest-xray-pneumonia.zip` from the link above (or use the Kaggle API
   after configuring `~/.kaggle/kaggle.json`):

   ```bash
   mkdir -p data
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data
   unzip data/chest-xray-pneumonia.zip -d data
   ```

3. Set `data_dir` in `config.yaml` to the directory containing `train/`, `val/`,
   and `test/`. With the standard archive this is `data/chest_xray`.

The same steps are available as a script:

```bash
bash scripts/download_kaggle_data.sh
```

Each split must contain `NORMAL/` and `PNEUMONIA/` directories. Do not put
patient images or Kaggle credentials in git; `data/` is ignored.

## Train, evaluate, and explain

```bash
python -m src.train --config config.yaml
python -m src.evaluate --config config.yaml
python -m src.explain \
  --image data/chest_xray/test/PNEUMONIA/person1_bacteria_1.jpeg \
  --checkpoint artifacts/best.pt \
  --output artifacts/attention-example.png
```

Training writes `artifacts/best.pt` and `artifacts/history.json`; evaluation
writes `artifacts/test_metrics.json`. CPU training is supported but can be slow.
Use `num_workers: 0` on systems where multiprocessing is restricted.

## Architecture

For an input image `x`, convolutional blocks produce feature tensor
`F ∈ R^(C×H×W)`. The attention layer computes:

```
F_avg(i,j) = (1/C) Σ_c F(c,i,j)
F_max(i,j) = max_c F(c,i,j)
A = sigmoid(Conv7×7([F_avg, F_max]))
F_att = F ⊙ A
z = GAP(F_att)
logits = W · Dropout(z) + b
```

For 224×224 input, four 2×2 pooling operations produce a 14×14 attention map.
`explain.py` bilinearly upsamples `A` to the original image size for a
qualitative overlay. Attention is an indication of where this model used
features; it is not a medically validated lesion segmentation.

## Metrics and experiment hygiene

The positive class is `PNEUMONIA`. Because class and patient distributions can
be imbalanced, inspect sensitivity and specificity alongside accuracy. The
provided split is used as supplied; for a serious study, verify patient-level
separation, tune a decision threshold on validation data, report confidence
intervals, and compare against a non-attention baseline. Keep generated metrics
with the exact config, seed, data version, and checkpoint.

## Limitations and ethical note

This is an educational research prototype, not a diagnostic device. Chest
radiographs may contain demographic, scanner, hospital, acquisition, and
labeling biases; the Kaggle labels and split provenance should be independently
audited. Performance on this dataset does not establish safety, fairness,
clinical utility, or generalization to a hospital. Attention heatmaps can be
misleading and must not be interpreted as explanations or evidence of disease.
Never use this model to diagnose, triage, or make treatment decisions. Any
clinical deployment requires representative external validation, specialist
oversight, privacy safeguards, regulatory review, and prospective monitoring.
