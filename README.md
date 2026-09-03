# PAAS_SeLop — SeLop / LROR for face forgery detection

Standalone implementation of **SeLop** (*Low-rank Orthogonal Subspace
Intervention for Generalizable Face Forgery Detection*, arXiv:2601.11915v2),
trained on the MIDS dataset.

## What it is

A **frozen CLIP ViT-L/14-336** backbone with a tiny trainable **LROR** module
(Low-rank Orthogonal Removal of spurious correlation) inserted before each of the
last 12 transformer blocks, plus a linear classification head. Total trainable
parameters ≈ **0.40M** (the CLIP backbone is fully frozen).

LROR learns, per layer, a skinny matrix `M ∈ R^{1024×r}`; its orthonormal basis
`Q = QR(M)` spans an estimated *spurious* subspace (identity / background / style
shortcuts). The visual tokens are projected onto the orthogonal complement:

```
X_vis  ←  X_vis (I − Q Qᵀ)          # [CLS] token is excluded, passes through untouched
```

forcing the classifier to key on forgery traces rather than identity/style cues —
the mechanism that drives cross-dataset generalization. Orthogonality is
structural (guaranteed by QR), so training uses **plain cross-entropy** with no
auxiliary or orthogonality loss.

## Labels (3-class, like the GSD project)

Ground-truth labels are derived from the image **path** via `get_label_all`
(vendored in `selop/get_label.py`), not from the json fields:

| class | id | source labels |
|-------|----|---------------|
| real     | 0 | REAL |
| pad      | 1 | PAD, **MAKEUP** (folded in) |
| deepfake | 2 | DEEPFAKE |

`UNKNOWN` images are dropped. Set `num_classes: 2` in `config.json` for plain
binary real/fake instead. CE uses inverse-frequency class weights
(`class_weight: true`) to handle imbalance.

## Metrics

The headline metrics on the test set are **acc / auc / ap** (binary real-vs-fake,
fake = positive). Also reported: EER, per-class recall (real/pad/deepfake), and an
operating point — the max fake-recall achievable at real-recall ≥ 95%
(`fake_recall@real95`) with its threshold. `best.pt` is selected by `select_metric`
(default `auc`).

## Data

- **train**: `/datasets/work/vLLM/temp/testset/testset_mids/mids_first_half.json`
  (1,710,990 images)
- **val**:   `/datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json`
  (30,197 images)

A slim `<json>.selop_idx.3c.tsv` (path + label) is cached next to each json on
first run so the 1.5 GB train json is parsed only once.

## Environment

Uses **`python3.12`** (torch 2.11+cu130, transformers 4.37.2, sklearn) and the
shared `clip-vit-large-patch14-336` weights (symlinked under `base_models/`).
DDP is launched via `python3.12 -m torch.distributed.run` (the `torchrun` shim
points at a different python).

## Train (4 GPUs, ~50% CPU)

```bash
cd /datasets/work/vLLM/temp/PAAS_SeLop
GPUS=0,1,2,3 OMP_NUM_THREADS=12 ./run_train.sh
```

- `GPUS` — comma-separated device list (default `0,1,2,3`).
- `OMP_NUM_THREADS=12` × 4 processes = 48 of 96 cores (~50%, the "OMP" cap).
- All hyperparameters live in `config.json`; override any from the CLI, e.g.
  `./run_train.sh --epochs 5 --rank 36`.
- Smoke test: `./run_train.sh --limit_train 1024 --limit_val 1024 --epochs 1`.

Checkpoints (`best.pt`, `last.pt`, ~1.6 MB each — only LROR + head are saved),
`metrics_log.jsonl`, and `config.used.json` are written to `out_dir`
(`runs/selop_default`).

## Inference

Single image (default):

```bash
python3.12 infer.py --ckpt runs/selop_default/best.pt --image /path/to/face.jpg
# -> {"image": ..., "p_fake": 0.93, "threshold": 0.71, "verdict": "FAKE"}
```

Full evaluation on a json:

```bash
python3.12 infer.py --ckpt runs/selop_default/best.pt --eval        # config's val_data
python3.12 infer.py --ckpt runs/selop_default/best.pt --val other.json
```

The threshold defaults to the checkpoint's stored `threshold@real95`; override
with `--threshold`.

## Default config (tuned for best accuracy, per the paper)

| field | value |
|-------|-------|
| backbone | frozen CLIP ViT-L/14-336 |
| LROR rank `r` | 32 (paper sweep 28/32/36) |
| intervention layers | last 12 of 24 |
| optimizer | Adam, lr 2e-4 (constant after 200-step warmup), wd 5e-4 |
| batch size | 32 / GPU (×4 = 128 effective) |
| precision | bf16 autocast (QR computed in fp32) |
| epochs | 3 |
| num_classes | 3 (real/pad/deepfake) |

## Files

```
config.json          # all defaults
run_train.sh         # DDP launcher (GPUs + OMP cap)
train.py             # DDP training loop
infer.py             # single-image + full-eval inference
selop/lror.py        # the LROR module
selop/model.py       # frozen CLIP + LROR + head
selop/data.py        # MIDS dataset, get_label_all -> 3-class, index cache
selop/engine.py      # distributed evaluation
selop/metrics.py     # acc/auc/ap + EER + per-class recall + operating point
selop/get_label.py   # vendored path->label (get_label_all)
selop/utils.py       # dist / seeding / CPU-thread cap
```
