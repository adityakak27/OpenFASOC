# Complete GRPO Training Pipeline - Data Flow

## Overview
This pipeline trains CodeLlama models in three stages:
1. **Initial Finetuning**: Domain adaptation on OpenFASOC code
2. **Self-Improvement**: Iterative training with master model feedback
3. **GRPO Optimization**: Reinforcement learning on DRC/LVS metrics

---

## 🔵 STAGE 1: Initial Finetuning (Iteration 0)

### Purpose
Teach the base CodeLlama model to understand OpenFASOC's analog circuit generator code patterns.

### Data Flow
```
external/OpenFASOC/*.py
    ↓ (process_data.py reads ALL .py files)
    ↓ (creates training samples from raw Python code)
processed_data/train_data_7b.jsonl
    ↓ (train.py with LoRA)
initial_finetuned_models/7b_<timestamp>/merged/
```

### Detailed Steps

**Step 1.1: Data Processing**
- Script: `scripts/others/process_data.py`
- Input: `external/OpenFASOC/**/*.py` (raw Python files)
- Process:
  - Scans all Python files in OpenFASOC directory
  - Filters out files < 100 characters
  - Deduplicates identical code blocks
  - Converts to training format:
    ```json
    {
      "filename": "voltage_divider.py",
      "analysis": [{
        "issue": "Code documentation and understanding",
        "explanation": {
          "problem": "Understanding OpenFASOC analog generator code patterns",
          "reason": "Learning domain-specific code structure and conventions",
          "fix": "Study and reproduce the code structure"
        }
      }],
      "fixed_code": "<actual OpenFASOC Python code>"
    }
    ```
- Output: `data/processed_initial_ft_<timestamp>/7b/train_data_7b.jsonl`

**Step 1.2: LoRA Training**
- Script: `scripts/training/train.py`
- Input: The JSONL file from Step 1.1
- Model: Base CodeLlama (`codellama/CodeLlama-7b-Instruct-hf`)
- Training:
  - LoRA (Low-Rank Adaptation) for efficient training
  - 3 epochs, learning rate 2e-4
  - Teaches model to reproduce OpenFASOC code style
- Output: `initial_finetuned_models/7b_<timestamp>/merged/`

**Result**: Model now understands OpenFASOC code patterns but hasn't learned to fix errors yet.

---

## 🟢 STAGE 2: Self-Improvement (Iterations 1, 2, 3...)

### Purpose
Iteratively improve the model's code generation by learning from master model (Gemini) feedback.

### Data Flow for Each Iteration
```
CurrentModel
    ↓ (generate code from prompts)
generated_code.py
    ↓ (compile and get errors)
compile_errors.txt
    ↓ (master model analyzes)
master_feedback.json
    ↓ (convert to training format)
iteration_N_training.jsonl
    ↓ (train.py with LoRA)
NextModel
```

### Detailed Steps Per Iteration

**Step 2.1: Base Inference** (only for iteration 1)
- Script: `scripts/inference/run_base_inferences.py`
- Model: Previous iteration's model
- Input: `scripts/utils/prompt.txt` (circuit design prompts)
- Process: Generate 10 Python files for different analog circuits
- Output: `pipeline_runs/<run_id>/outputs/iteration_N/base/*.py`

**Step 2.2: Fine-tune Model**
- Script: `scripts/training/train.py`
- Model: Previous iteration's model
- Input: `pipeline_runs/<run_id>/training_data/iteration_<N-1>_training.jsonl`
- Training:
  - Uses master model's structured feedback from PREVIOUS iteration
  - LoRA training, 1 epoch, learning rate 1e-6
  - Learns to fix specific issues identified by master model
- Output: `pipeline_runs/<run_id>/models/iteration_N/merged/`

**Step 2.3: Fine-tuned Inference**
- Script: `scripts/inference/run_base_inferences.py`
- Model: Newly fine-tuned model from Step 2.2
- Input: Same prompts as Step 2.1
- Process: Generate 10 Python files with improved model
- Output: `pipeline_runs/<run_id>/outputs/iteration_N/finetuned/*.py`

**Step 2.4-5: Compile Outputs**
- Script: `scripts/compile/compile_generated_code.py`
- Input: Generated Python files from Step 2.3
- Process:
  - Run `python -m py_compile` on each file
  - Capture compilation errors, warnings, syntax issues
  - Save error messages to text files
- Output: `pipeline_runs/<run_id>/compile_results/iteration_N/*.txt`

**Step 2.6: Master Model Inference**
- Script: `scripts/inference/master_model_inference.py`
- Model: **Gemini (master model)** - NOT the student model
- Input:
  - Generated Python files from Step 2.3
  - Compilation errors from Step 2.4-5
  - Original prompts
- Process:
  - Master model analyzes each generated file
  - Identifies specific issues (syntax, logic, missing imports, etc.)
  - Provides structured feedback:
    ```json
    {
      "filename": "voltage_divider.py",
      "analysis": [
        {
          "issue": "Missing import statement",
          "explanation": {
            "problem": "gdsfactory module is used but not imported",
            "reason": "Will cause NameError at runtime",
            "fix": "Add 'import gdsfactory as gf' at top of file"
          }
        },
        {
          "issue": "Incorrect parameter type",
          "explanation": {
            "problem": "width parameter should be float, not string",
            "reason": "Layout functions expect numeric values",
            "fix": "Convert width='5' to width=5.0"
          }
        }
      ],
      "fixed_code": "<corrected version of the code>"
    }
    ```
- Output: `pipeline_runs/<run_id>/master_predictions/iteration_N/*.txt`

**Step 2.7: Convert to Training Data**
- Script: `scripts/utils/convert_training_data.py`
- Input: Master model JSON outputs from Step 2.6
- Process:
  - Extract JSON from master model's text output
  - Validate structure (must have filename, analysis, fixed_code)
  - Clean and deduplicate
- Output: `pipeline_runs/<run_id>/training_data/iteration_N_training.jsonl`

**Result**: This training data is used in the NEXT iteration's Step 2.2!

### Key Insight
Each iteration's training uses the PREVIOUS iteration's master model feedback. The cycle:
```
Iteration N generates code
    ↓
Master evaluates code
    ↓
Create training data from evaluation
    ↓
Iteration N+1 trains on this data
    ↓
Iteration N+1 generates better code
```

---

## 🔴 STAGE 3: GRPO Optimization

### Purpose
Fine-tune the model using reinforcement learning to optimize for actual DRC/LVS metrics (not just code quality).

### Data Flow
```
output.json (DRC/LVS results)
    ↓ (parse errors and metrics)
error_prompts.json
    ↓ (GRPO training with rewards)
    ↓ (reward = -drc_errors - lvs_errors)
grpo_trained_model/
```

### Detailed Steps

**Step 3.1: Load Evaluation Data**
- Script: `grpo_pipeline.py`
- Input: `output.json` (contains DRC/LVS test results for generated circuits)
- Format:
  ```json
  {
    "component_name": "voltage_divider",
    "drc_result": {
      "passed": false,
      "errors": 5,
      "error_details": ["Metal spacing violation", ...]
    },
    "lvs_result": {
      "passed": true,
      "errors": 0
    },
    "original_code": "...",
    "parameters": {...}
  }
  ```

**Step 3.2: Extract Errors**
- Identify samples with DRC or LVS failures
- Create error samples with full context

**Step 3.3: GRPO Training**
- Script: `grpo.py` (grpo_trainer class)
- Model: Self-improved model from Stage 2
- Algorithm: **Group Relative Policy Optimization**
- Process:
  - Generate multiple code variations for each prompt
  - Evaluate each with DRC/LVS (reward = -errors)
  - Compute group-relative advantages
  - Update policy to favor high-reward outputs
  - KL penalty keeps model close to reference
- Training:
  - 10 epochs
  - Batch size 4 (group size for GRPO)
  - LoRA for efficiency
- Output: `grpo_outputs/<model_size>_<timestamp>/iteration_N/trained_model/`

**Step 3.4: Generate Error Prompts**
- Create specific prompts for fixing DRC/LVS errors
- Include error context and original code

**Step 3.5: Generate Fixed Code**
- Use GRPO-trained model to generate fixes
- Validate improvements

**Result**: Model optimized to generate circuits that pass DRC/LVS checks.

---

## Summary: What Data Comes From Where

| Stage | Data Source | Purpose |
|-------|-------------|---------|
| **Initial FT** | `external/OpenFASOC/*.py` | Learn domain code patterns |
| **Self-Improve Iter 1** | Master evaluation of initial model's output | Learn to fix issues |
| **Self-Improve Iter 2** | Master evaluation of iter 1 model's output | Learn from improved attempts |
| **Self-Improve Iter 3** | Master evaluation of iter 2 model's output | Continue refinement |
| **GRPO** | `output.json` (DRC/LVS metrics) | Optimize for actual circuit metrics |

## Key Principles

✅ **No circular data**: Each iteration uses feedback from the PREVIOUS iteration, not its own
✅ **No "data" section code**: Initial training ONLY uses `external/OpenFASOC/`
✅ **Progressive improvement**: Model → Generate → Evaluate → Train → Better Model
✅ **External evaluation**: Master model (Gemini) provides independent feedback
✅ **Metric-driven final stage**: GRPO uses real DRC/LVS scores, not subjective quality

## File Structure
```
llm-finetuning-folder/
├── external/OpenFASOC/          # Source data (ONLY for initial training)
├── initial_finetuned_models/    # Stage 1 output
├── pipeline_runs/
│   └── <run_id>/
│       ├── models/              # Stage 2 models per iteration
│       ├── outputs/             # Generated code per iteration
│       ├── compile_results/     # Compilation errors
│       ├── master_predictions/  # Master model feedback
│       └── training_data/       # Training data for next iteration
└── grpo_outputs/                # Stage 3 output
```

