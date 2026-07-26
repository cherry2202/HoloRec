# HoloRec

**Holistic Encoding and Interleaved Reasoning for Generative Recommendation**

HoloRec is a generative recommendation framework that couples hierarchical item
representations with endogenous coarse-to-fine reasoning. Instead of treating
semantic IDs as a single flat sequence, HoloRec constructs multi-granularity
codes and uses their coarse semantics to guide fine-grained item generation.

## Overview

HoloRec contains three progressively integrated components:

1. **Holistic semantic encoding (HoloSID)** uses multi-granularity nested
   residual quantization and a holistic reconstruction objective to produce
   semantically meaningful item codes at multiple resolutions.
2. **Non-thinking mode (HoloRec-NT)** adds coarse-grained auxiliary supervision
   during training while retaining fine-only autoregressive decoding at
   inference time.
3. **Interleaved reasoning** predicts coarse semantics during decoding and
   injects them into the fine-grained decoder through a gated residual path.

The framework uses T5 as its generative backbone and is implemented in PyTorch.
The experiments currently cover the Amazon **Beauty** and **Musical
Instruments** recommendation datasets.

## Main results

| Dataset | Model | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| Beauty | HoloRec | 0.0451 | 0.0683 | 0.0307 | 0.0375 |
| Instruments | HoloRec | 0.0894 | 0.1105 | 0.0761 | 0.0828 |

HoloRec improves over the strongest generative baseline most clearly on the
sparser Beauty dataset. See the paper and files under `rst/` for the complete
comparison, ablations, and prediction outputs.

## Repository layout

```text
.
├── data/                                  # Interactions, item metadata, and semantic IDs
├── rst/                                   # Evaluation and ablation results
├── efficiency_benchmark.py                # Interleaved-inference GPU benchmark
├── efficiency_benchmark_non_thinking.py   # Non-thinking GPU benchmark
├── test_efficiency_non_thinking.sh        # Example non-thinking benchmark command
└── EFFICIENCY_EXPERIMENT.md               # Reproducible efficiency protocol
```

The benchmark scripts are designed to run alongside the core HoloRec training
and evaluation modules (`collator.py`, `modeling_letter.py`, `test.py`, and
`utils.py`). Checkpoints and raw Amazon review data are not committed.

## Environment

A CUDA-capable GPU is recommended. The experiments in the paper were run on
NVIDIA A100 GPUs.

Core dependencies include:

- Python 3
- PyTorch with CUDA support
- Hugging Face Transformers
- SentencePiece

Install the versions required by your CUDA environment, for example:

```bash
pip install torch transformers sentencepiece
```

## Data preparation

Place each dataset's interaction file, item metadata, and semantic-ID files in
`data/`. The included Beauty example follows this naming convention:

```text
data/
├── Beauty.inter.json
├── Beauty.item.json
├── Beauty.tw8.json
└── Beauty.tw32.json
```

Here, `tw8` and `tw32` denote coarse- and fine-grained HoloSID variants. Keep
the corresponding suffixes consistent when passing `--coarse_index_file` and
`--index_file` to the training, evaluation, or benchmark scripts.

## Efficiency benchmark

`efficiency_benchmark.py` measures generation only. Checkpoint loading, data
loading, tokenization, metric computation, and result serialization are outside
the timed region.

Run a short smoke test before collecting results:

```bash
export CUDA_VISIBLE_DEVICES=0

python efficiency_benchmark.py \
  --ckpt_path /path/to/checkpoint \
  --dataset Beauty \
  --data_path ./data \
  --index_file .tw32.json \
  --coarse_index_file .tw8.json \
  --test_batch_size 1 \
  --num_beams 20 \
  --precision bf16 \
  --warmup_batches 2 \
  --measure_batches 3 \
  --repeats 2 \
  --output_json results/efficiency/smoke.json
```

For publishable measurements, use the full protocol in
[`EFFICIENCY_EXPERIMENT.md`](EFFICIENCY_EXPERIMENT.md). It reports latency,
throughput, peak CUDA memory, parameter counts, and the hardware/software
environment in JSON, with optional per-repeat CSV output.

## Reproducibility notes

- Use the same semantic-ID files, checkpoint, beam width, precision, and history
  limit when comparing model variants.
- Use one GPU and one process for per-model latency measurements.
- Preserve raw JSON/CSV outputs and report the mean and standard deviation over
  repeated runs.
- Do not compare measurements collected on different GPU models as though they
  were directly equivalent.

## Citation

If you find this project useful, please cite the paper. The BibTeX entry will be
added after publication.

```bibtex
@inproceedings{holorec,
  title  = {HoloRec: Holistic Encoding and Interleaved Reasoning for Generative Recommendation},
  author = {Anonymous},
  note   = {Under review}
}
```

## License

No license has been specified yet. Please contact the repository owner before
redistributing or using the code in another project.
