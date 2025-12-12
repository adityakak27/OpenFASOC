# VLSI Parameter Generation with GRPO - Batch Training System

## 📋 Overview

This repository contains a **GRPO (Group Relative Policy Optimization)** trained model for generating VLSI Transmission Gate parameters. The model is based on **CodeLlama-7b-Instruct** and has been fine-tuned to output circuit parameters in JSON format.

### Current State (as of Dec 1, 2025)

- **Trained Model**: `grpo_standalone_outputs/run_20251130_062650/trained_model/merged/`
- **Base Model**: `codellama/CodeLlama-7b-Instruct-hf`
- **Training Data**: 3000 samples from `output.json` (1500 pass DRC+LVS)
- **Training Duration**: ~12.5 hours on H100 GPU
- **Current Issue**: Model outputs valid JSON parameters but doesn't optimize for metrics (area, resistance, etc.)

---

## 🎯 The Goal: Batch Training with Real Evaluator Feedback

The current training used **proxy rewards** (text pattern matching) instead of **real circuit metrics**. To make the model actually learn to generate better circuits, we need to implement a **batch training loop** with your evaluator box.

### Proposed Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         BATCH TRAINING LOOP                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────┐                                                          │
│  │  THIS MACHINE   │                                                          │
│  │  (GPU Server)   │                                                          │
│  └────────┬────────┘                                                          │
│           │                                                                   │
│           ▼                                                                   │
│  ┌─────────────────────────────────────────┐                                  │
│  │ Step 1: Generate N Parameter Sets       │                                  │
│  │ - Load trained model                    │                                  │
│  │ - Run inference N times (e.g., 1000)    │                                  │
│  │ - Save to: generated_params_batch_X.json│                                  │
│  └────────┬────────────────────────────────┘                                  │
│           │                                                                   │
│           │ Transfer JSON file                                                │
│           ▼                                                                   │
│  ┌─────────────────────────────────────────┐                                  │
│  │ OTHER MACHINE (Evaluator Box)           │                                  │
│  │                                         │                                  │
│  │ Step 2: Run Evaluation                  │                                  │
│  │ - For each param set:                   │                                  │
│  │   - transmission_gate.py generates GDS  │                                  │
│  │   - DRC check → pass/fail + errors      │                                  │
│  │   - LVS check → pass/fail + mismatches  │                                  │
│  │   - PEX extraction → R, C values        │                                  │
│  │   - Geometric analysis → area, symmetry │                                  │
│  │ - Save to: eval_results_batch_X.json    │                                  │
│  └────────┬────────────────────────────────┘                                  │
│           │                                                                   │
│           │ Transfer JSON file back                                           │
│           ▼                                                                   │
│  ┌─────────────────────────────────────────┐                                  │
│  │ Step 3: Compute Rewards from Real Metrics│                                 │
│  │                                         │                                  │
│  │ reward = (                              │                                  │
│  │   +10.0 if drc_pass else -5.0           │                                  │
│  │   +10.0 if lvs_pass else -10.0          │                                  │
│  │   +5.0 * (1 - area/1000) if area<1000   │  ← Real circuit metrics!        │
│  │   +3.0 * symmetry_score                 │                                  │
│  │   -0.001 * resistance                   │                                  │
│  │ )                                       │                                  │
│  └────────┬────────────────────────────────┘                                  │
│           │                                                                   │
│           ▼                                                                   │
│  ┌─────────────────────────────────────────┐                                  │
│  │ Step 4: GRPO Training Update            │                                  │
│  │ - Group completions by prompt           │                                  │
│  │ - Compute advantages from real rewards  │                                  │
│  │ - Update model weights                  │                                  │
│  │ - Save checkpoint                       │                                  │
│  └────────┬────────────────────────────────┘                                  │
│           │                                                                   │
│           │ Repeat until convergence                                          │
│           ▼                                                                   │
│  ┌─────────────────────────────────────────┐                                  │
│  │ Step 5: Evaluate Progress               │                                  │
│  │ - Track: DRC pass rate, LVS pass rate   │                                  │
│  │ - Track: Average area, resistance       │                                  │
│  │ - Save best model checkpoint            │                                  │
│  └─────────────────────────────────────────┘                                  │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
llm-finetuning-folder/
├── README_BATCH_TRAINING.md    # This file - detailed implementation guide
├── requirements.txt            # Python dependencies
├── env.yml                     # Conda environment specification
│
├── grpo.py                     # Core GRPO trainer with reward functions
├── run_grpo.py                 # Standalone training script
├── eval_inference.py           # Inference script for generating parameters
│
├── output.json                 # Training data (3000 samples with metrics)
│
├── grpo_standalone_outputs/    # Training outputs
│   └── run_20251130_062650/
│       ├── trained_model/
│       │   ├── lora_adapter/   # LoRA weights only
│       │   └── merged/         # Full merged model (USE THIS)
│       └── training_data.json
│
└── batch_training/             # [TO BE CREATED] Batch training scripts
    ├── generate_batch.py       # Generate N parameter sets
    ├── process_eval_results.py # Convert eval results to rewards
    └── train_on_batch.py       # Train model on batch with real rewards
```

---

## 🔧 Implementation Details

### 1. Parameter Generation Format

The model generates parameters for Transmission Gates with this structure:

```json
{
  "width": [NMOS_width, PMOS_width],      // μm, range: 0.5-20.0
  "length": [NMOS_length, PMOS_length],   // μm, range: 0.15-4.0
  "fingers": [NMOS_fingers, PMOS_fingers], // integer, range: 1-8
  "multipliers": [NMOS_mult, PMOS_mult]   // integer, typically [1,1]
}
```

**Optimal parameters** (from training data analysis):
- Area < 500 μm²: `width` 0.5-3.0, `fingers` [1,1] or [2,2], `multipliers` [1,1]
- LVS requires exactly 6 nets, 4 devices

### 2. Evaluator Output Format (Expected from your eval box)

```json
{
  "sample_id": "tg_sample_1000",
  "parameters": { ... },
  "component_name": "Transmission_Gate",
  "drc": {
    "is_pass": true,
    "summary": {
      "total_errors": 0,
      "critical_errors_count": 0,
      "reviewable_errors_count": 0,
      "passable_errors_count": 0
    }
  },
  "lvs": {
    "is_pass": true,
    "summary": {
      "conclusion": "LVS Pass: Netlists match.",
      "mismatch_details": {
        "nets": "6 | Number of nets: 6",
        "devices": "4 | Number of devices: 4"
      }
    }
  },
  "geometric": {
    "raw_area_um2": 450.5,
    "symmetry_score_horizontal": 1.0,
    "symmetry_score_vertical": 0.85
  },
  "pex": {
    "total_resistance_ohms": 50000.0,
    "total_capacitance_farads": 1e-14
  }
}
```

### 3. Reward Function Design

```python
def compute_real_reward(eval_result: dict) -> float:
    """
    Compute reward from REAL evaluator metrics.
    This replaces the proxy text-based rewards.
    """
    reward = 0.0
    
    # DRC compliance (most important)
    if eval_result['drc']['is_pass']:
        reward += 10.0
    else:
        # Penalty based on error severity
        summary = eval_result['drc']['summary']
        reward -= 2.0 * summary.get('critical_errors_count', 0)
        reward -= 1.0 * summary.get('reviewable_errors_count', 0)
        reward -= 0.5 * summary.get('passable_errors_count', 0)
    
    # LVS compliance (critical)
    if eval_result['lvs']['is_pass']:
        reward += 10.0
    else:
        reward -= 10.0  # LVS fail is very bad
    
    # Area optimization (smaller is better)
    area = eval_result['geometric']['raw_area_um2']
    if area < 500:
        reward += 5.0 * (1 - area / 500)  # Max +5 for smallest area
    elif area < 1000:
        reward += 2.0 * (1 - area / 1000)
    else:
        reward -= 0.001 * (area - 1000)  # Penalty for large area
    
    # Symmetry bonus
    h_sym = eval_result['geometric']['symmetry_score_horizontal']
    v_sym = eval_result['geometric']['symmetry_score_vertical']
    reward += 2.0 * (h_sym + v_sym) / 2
    
    # PEX: Low resistance is good (but not zero)
    resistance = eval_result['pex']['total_resistance_ohms']
    if 1000 < resistance < 100000:
        reward += 2.0
    elif resistance > 500000:
        reward -= 1.0
    
    return reward
```

### 4. Key Training Parameters

From the successful training run:
```python
# Model configuration
base_model = "codellama/CodeLlama-7b-Instruct-hf"
max_seq_length = 2048
load_in_4bit = True

# LoRA configuration
lora_r = 64
lora_alpha = 64
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]

# GRPO configuration
learning_rate = 2e-5
group_size = 4  # Number of completions per prompt
kl_coef = 0.1   # KL divergence coefficient
batch_size = 4  # Gradient accumulation steps
```

---

## 🚀 How to Implement Batch Training

### Step 1: Generate Parameter Batch (GPU Machine)

```python
# generate_batch.py
from unsloth import FastLanguageModel
import json

MODEL_PATH = "grpo_standalone_outputs/run_20251130_062650/trained_model/merged"
NUM_SAMPLES = 1000

def generate_batch():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    samples = []
    for i in range(NUM_SAMPLES):
        # Generate parameters (see eval_inference.py for full implementation)
        params = generate_single_sample(model, tokenizer, sample_id=i)
        samples.append(params)
    
    with open(f'batch_{batch_id}_params.json', 'w') as f:
        json.dump(samples, f, indent=2)
```

### Step 2: Evaluate on Eval Box (Other Machine)

```bash
# On evaluator machine
python run_batch_evaluation.py \
    --input batch_0_params.json \
    --output batch_0_results.json
```

### Step 3: Train with Real Rewards (GPU Machine)

```python
# train_on_batch.py
def train_on_evaluated_batch(params_file, results_file):
    # Load parameters and evaluation results
    with open(params_file) as f:
        params = json.load(f)
    with open(results_file) as f:
        results = json.load(f)
    
    # Compute real rewards
    rewards = [compute_real_reward(r) for r in results]
    
    # Create training dataset with (prompt, completion, reward) tuples
    training_data = []
    for p, r, reward in zip(prompts, completions, rewards):
        training_data.append({
            'prompt': p,
            'completion': r,
            'reward': reward
        })
    
    # Run GRPO update (see grpo.py for trainer implementation)
    trainer.train_on_rewards(training_data)
```

---

## ⚠️ Known Issues & Solutions

### Issue 1: Reward Saturation (reward_std = 0)
**Problem**: All completions get the same reward, no learning signal.
**Solution**: Use real evaluator metrics which will naturally have variance.

### Issue 2: Model Generates Invalid Parameters
**Problem**: Parameters outside valid ranges or wrong array lengths.
**Solution**: Add validation layer (see `validate_and_fix_params()` in `eval_inference.py`)

### Issue 3: Model Doesn't Optimize for Area
**Problem**: GRPO rewards didn't include area in the reward function.
**Solution**: Include `area_um2` directly in reward calculation from evaluator.

---

## 📊 Training Data Analysis

From `output.json` (3000 samples, 1500 pass DRC+LVS):

| Parameter | Min | Max | Avg | Best for Low Area |
|-----------|-----|-----|-----|-------------------|
| NMOS Width | 1.13 | 19.72 | 9.73 | 0.5-3.0 |
| PMOS Width | 0.58 | 19.63 | 8.85 | 0.5-2.5 |
| NMOS Length | 0.19 | 3.98 | 1.75 | 0.5-2.0 |
| PMOS Length | 0.15 | 3.99 | 2.07 | 0.2-2.0 |
| Fingers | 1-5 | 1-5 | - | [1,1] or [2,2] |
| Multipliers | 1 | 1 | 1 | [1,1] always |
| **Area (μm²)** | 336 | 3788 | 1557 | < 500 |

---

## 🔄 Migration Checklist

When moving to the evaluator machine:

1. [ ] Copy this entire folder
2. [ ] Install conda environment: `conda env create -f env.yml`
3. [ ] Or install via pip: `pip install -r requirements.txt`
4. [ ] Verify model loads: `python -c "from unsloth import FastLanguageModel; print('OK')"`
5. [ ] Test inference: `python eval_inference.py`
6. [ ] Implement `run_batch_evaluation.py` that calls your evaluator
7. [ ] Implement the batch training loop

---

## 📞 Contact / Handoff Notes

**Last worked on**: December 1, 2025
**GPU Used**: NVIDIA H100 80GB
**Training Time**: ~12.5 hours for 935 steps
**Key Files**:
- `grpo.py`: Core trainer, reward functions (lines 93-424)
- `eval_inference.py`: Inference with validation (lines 1-200)
- `output.json`: Full training dataset with metrics

**What the next person needs to do**:
1. Create the evaluator interface script
2. Implement batch generation → evaluation → training loop
3. Monitor real metrics (DRC/LVS pass rate, area) during training
4. Iterate until model consistently produces low-area, passing circuits

---

## 📝 License

Internal use only. Based on CodeLlama (Meta AI License) and Unsloth (Apache 2.0).


