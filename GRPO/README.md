# Complete GRPO Training Pipeline

End-to-end training pipeline for VLSI circuit generation models from base CodeLlama models to GRPO-optimized models.

## Pipeline Overview

The complete pipeline consists of three main phases:

### **Phase 1: Initial Finetuning**
- Starts from base CodeLlama models (7B or 13B)
- Processes OpenFASOC circuit design data
- Applies LoRA finetuning for efficient domain adaptation
- Output: Domain-adapted model

### **Phase 2: Self-Improvement with Master Model**
- Model generates code predictions
- Code is compiled and tested
- Master model (Gemini) analyzes code quality and provides feedback
- Model is fine-tuned on master feedback
- Iterates for 3 rounds
- Output: Self-improved model

### **Phase 3: GRPO Optimization**
- Uses DRC/LVS evaluation metrics from `output.json`
- Applies Group Relative Policy Optimization
- Optimizes for design rule compliance and circuit correctness
- Iterates for 3 rounds
- Output: Final GRPO-optimized model

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (24GB+ VRAM recommended for 7B, 40GB+ for 13B)
- Google API key for Gemini master model evaluation
- DRC/LVS evaluation data in `output.json`

### Installation

```bash
pip install -r requirements.txt
```

### Run Pipeline

```bash
# Make script executable (first time only)
chmod +x run_pipeline.sh

# Run for 7B model
./run_pipeline.sh 7b

# Run for 13B model
./run_pipeline.sh 13b

# Run for both models sequentially
./run_pipeline.sh both
```

### Interactive Prompts

The script will prompt you for:

1. **Master Model Selection** - Choose Gemini model for code evaluation:
   ```
   [1] gemini-2.0-flash-exp   (Fast, recommended)
   [2] gemini-2.5-pro         (Best quality, slower)
   [3] gemini-1.5-pro         (Stable production)
   [4] gemini-1.5-flash       (Fast production)
   [5] gemini-pro             (Legacy)
   ```

2. **Google API Key** - Enter your Gemini API key
   - Get your key from: https://makersuite.google.com/app/apikey
   - Or set environment variable: `export GOOGLE_API_KEY="your_key"`

3. **Confirmation** - Press Enter to start the pipeline

## Output Structure

```
initial_finetuned_models/
└── {model_size}_{timestamp}/
    └── CodeLlama-{size}-Instruct-hf/
        ├── adapter_model.safetensors    # LoRA adapter
        └── merged/                      # Full finetuned model

pipeline_runs/{timestamp}/
├── models/
│   ├── iteration_0/                     # First self-improvement
│   ├── iteration_1/                     # Second self-improvement
│   └── iteration_2/                     # Final self-improved model
│       └── CodeLlama-{size}-Instruct-hf/
│           └── merged/                  # Use for GRPO
├── master_predictions/                  # Gemini code analysis
├── training_data/                       # Generated training data
└── plots/                               # Performance metrics

grpo_outputs/{model_size}_complete_{timestamp}/
├── iteration_0/
├── iteration_1/
└── iteration_2/
    └── trained_model/
        ├── adapter_model.safetensors    # GRPO LoRA adapter
        └── merged/                      # ← FINAL OPTIMIZED MODEL
```

## Pipeline Steps in Detail

### Step 1: Initial Finetuning (~2-4 hours)

```bash
# Automatically runs:
# 1. Process OpenFASOC data (circuit designs, routing, placement)
# 2. LoRA finetuning with:
#    - Learning rate: 2e-4
#    - LoRA rank: 16
#    - LoRA alpha: 32
#    - Epochs: 3
```

**Output**: `initial_finetuned_models/{model_size}_{timestamp}/*/merged/`

### Step 2: Self-Improvement (~4-8 hours)

```bash
# Automatically runs 3 iterations of:
# 1. Model generates circuit code
# 2. Code is compiled
# 3. Gemini analyzes code quality
# 4. Model is finetuned on feedback
```

**Output**: `pipeline_runs/{timestamp}/models/iteration_2/*/merged/`

### Step 3: GRPO Optimization (~6-12 hours)

```bash
# Automatically runs 3 iterations of:
# 1. Generate circuits
# 2. Evaluate DRC/LVS metrics
# 3. Calculate rewards/penalties
# 4. Update model with GRPO
```

**Output**: `grpo_outputs/{model_size}_complete_{timestamp}/iteration_2/trained_model/merged/`

## Expected Timeline

| Model Size | Total Time | GPU Memory |
|------------|------------|------------|
| 7B         | 12-16 hours | 24GB+     |
| 13B        | 18-24 hours | 40GB+     |
| Both       | 30-40 hours | 40GB+     |

## Monitoring Progress

### Watch logs in real-time

```bash
# For self-improvement pipeline
tail -f pipeline_runs/*/pipeline.log

# For GRPO training
tail -f grpo_outputs/*/logs/grpo_pipeline.log
```

### Check GPU usage

```bash
watch nvidia-smi
```

## Using the Final Model

After the pipeline completes, use the final GRPO-optimized model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the final model
model_path = "grpo_outputs/7b_complete_{timestamp}/iteration_2/trained_model/merged"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Generate circuit code
prompt = "Create a voltage follower circuit using glayout..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.1)
code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(code)
```

## Troubleshooting

### Issue: Out of memory during training

**Solution**: Try 7B model instead of 13B, or reduce batch size in the scripts

### Issue: "GOOGLE_API_KEY not provided"

**Solution**: 
```bash
export GOOGLE_API_KEY="your_api_key_here"
./run_pipeline.sh 7b
```

### Issue: Master model API rate limit

**Solution**: The script includes automatic retry logic. If persistent, reduce `--inferences_per_iteration` in the script.

### Issue: Training interrupted

**Solution**: The pipeline saves checkpoints. You can manually resume from the last completed phase by:
1. Check which phase completed (look at output directories)
2. Run the remaining phases manually using the Python scripts

## Manual Pipeline Execution

If you need more control, run each phase manually:

### Phase 1: Initial Finetuning

```bash
# Process data
python scripts/others/process_data.py \
    --model 7b \
    --openfasoc_path ./external/OpenFASOC/ \
    --output_dir data/processed_manual

# Finetune
python scripts/training/train.py \
    --train_file data/processed_manual/train_data_7b.jsonl \
    --output_dir initial_ft_manual \
    --model_paths codellama/CodeLlama-7b-Instruct-hf \
    --use_lora \
    --num_train_epochs 3
```

### Phase 2: Self-Improvement

```bash
python end_to_end.py \
    --model_size 7b \
    --base_model initial_ft_manual/*/merged \
    --master_model gemini-2.0-flash-exp \
    --prompt_file scripts/utils/prompt.txt \
    --google_api_key YOUR_KEY \
    --iterations 3
```

### Phase 3: GRPO

```bash
python grpo_pipeline.py \
    --input_json output.json \
    --base_model pipeline_runs/*/models/iteration_2/*/merged \
    --output_dir grpo_manual \
    --num_epochs 10 \
    --iterations 3
```

## Requirements

```bash
torch>=2.0.0
transformers>=4.30.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.20.0
google-generativeai>=0.3.0
numpy
datasets
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{grpo-vlsi-2025,
  title={GRPO Training Pipeline for VLSI Circuit Generation},
  author={Your Name},
  year={2025}
}
```

## License

MIT License

## Support

For issues or questions, please open an issue on GitHub or contact the maintainers.
