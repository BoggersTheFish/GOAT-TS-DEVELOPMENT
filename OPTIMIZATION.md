# GOAT-TS-SUPERLITE: Optimization Guide

## Short-term (1–2 days)

- **Rust wave engine**: Implemented in `rust/src/wave_engine.rs` with decay, convergence (epsilon/max_ticks), and neighbor-based propagation.
- **Hypothesis chaining**: Implemented in `python/goat_ts_cig/hypothesis_engine.py` (chains by shared premise nodes).

## Medium-term (3–7 days)

- **LITE auto-modes**: RAM and core detection via `sysinfo`; tick cap and batch hint from `get_hardware_limits()`.
  - Call from Python: `from bindings.rust_bindings import get_hardware_limits; max_ticks, batch_hint = get_hardware_limits()`.
  - RAM &lt; 4 GB → max 25 ticks; &lt; 8 GB → 50; else 100. Batch hint scales with physical cores.
- **Zero-copy bindings**: Use `simulate_waves_flat_buf(node_ids, initial_influences_buf, edges, ...)` with a buffer (e.g. `numpy` array or `array.array`) for `initial_influences_buf` to avoid list conversion.
- **Interference (positive/negative waves)**: Edge `strength` can be negative; the Rust engine uses it as-is (inhibition / negative influence).

## Long-term: Profiling and low-core tuning

1. **Profile**
   - Python: `python -m timeit -n 5 -r 2 -s "from run import *" "run_cig('your seed')"` or use `py-spy`:  
     `py-spy top -- python run.py --seed "large topic"`
   - Rust: build with `maturin develop --release` for release profile; use `RAYON_NUM_THREADS=2` (or 4) to simulate low-core.

2. **Cursor prompt for Rust loop tuning**
   - Use this in Cursor (Composer/agent) after profiling:
   - *"Optimize this Rust loop for 2–4 core Pentium — use rayon work-stealing, reduce allocations. Focus on `wave_engine.rs` simulate_waves_flat: double-buffer is already in place; consider chunk size, avoiding extra Vec allocations, and rayon scope."*

3. **Test on low-end PC**
   - Run with larger seeds or more nodes: `python run.py --seed "AI ethics and policy"`.
   - Measure: **htop** (Linux/macOS) or **Task Manager** (Windows) — watch CPU cores and RAM.
   - Optional: set `RAYON_NUM_THREADS=2` (or 4) to limit threads and mimic 2–4 core.

## Quick benchmark (timeit)

From repo root (GOAT-TS-SUPERLITE):

```bash
python -m timeit -n 3 -r 2 -s "
import sys
from pathlib import Path
root = Path('.').resolve()
sys.path.insert(0, str(root / 'python'))
sys.path.insert(0, str(root))
from goat_ts_cig.interface import run_cig
" "run_cig('test')"
```

Or run the provided `benchmark.py` (see below).

---

## Run CLI (Phase 6)

From **GOAT-TS-SUPERLITE** (repo root):

```bash
python run.py --seed "AI ethics"
python run.py --text "Machine learning and AI ethics are important."
python run.py --file examples/raw_texts/sample.txt --json
```

Exactly one of `--seed`, `--text`, or `--file` is required; `--json` prints full JSON (idea_map, hypotheses).

## Data generation (instruction-tuning)

```bash
python python/generate_dataset.py --input examples/raw_texts -o data/finetune_dataset.jsonl
python python/generate_dataset.py -i file.txt -n 20 --format alpaca
```

## Fine-tuning a small LLM (LoRA)

```bash
pip install transformers peft datasets accelerate
```

Load `data/finetune_dataset.jsonl` with `datasets.load_dataset("json", data_files=..., split="train")`, then use **peft** `LoraConfig` + `get_peft_model` with a small causal LM (e.g. TinyLlama). See [Hugging Face PEFT](https://huggingface.co/docs/peft) and [Transformers](https://huggingface.co/docs/transformers).
