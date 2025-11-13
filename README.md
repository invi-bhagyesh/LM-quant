# Benchmarking SLMs using Quantization and RLHF (DPO)

This project benchmarks Small Language Models (SLMs) using **quantization** and **Reinforcement Learning from Human Feedback (RLHF)** through **Direct Preference Optimization (DPO)**. It uses the `lm-evaluation-harness` library for standardized evaluation and performance comparison across different models and datasets.

---

## Installation

Clone the repository and install dependencies:

```
git clone https://github.com/your-repo/qllm_eval.git
cd qllm_eval
git checkout anthropic
pip install -r requirements.txt
pip install lm-evaluation-harness
```
---

## Running Evaluation

Use the main evaluation script to benchmark quantized or DPO-trained models:
```
CUDA_VISIBLE_DEVICES=0 python qllm_eval/evaluation/q_harness/main.py 
    --model_path {model}
    --tasks {task}
    --w_bit {w_bit}
    --a_bit 16
    --w_group_size 128
    --kv_bit 16
```
---

## Examples

### Quantized Evaluation
```
CUDA_VISIBLE_DEVICES=0 python qllm_eval/evaluation/q_harness/main.py 
    --model_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    --tasks crows_pairs_english_religion
    --w_bit 8
```
### RLHF (DPO) Evaluation
```
CUDA_VISIBLE_DEVICES=0 python qllm_eval/evaluation/q_harness/main.py 
    --model_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    --tasks helpfulness
    --dataset PKU
    --dpo_path "checkpoints/tinyllama_dpo/"
```
---

## Notes

- Models are fetched from **Hugging Face**
- Built using **lm-evaluation-harness**
- Quantization reduces memory usage and improves efficiency
- DPO enhances model alignment after quantization

---

## Acknowledgement

Heavily based on `qllm-eval` and uses `lm-evaluation-harness` framework.
