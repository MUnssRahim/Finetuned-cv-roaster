# CV Roaster Model 

A specialized, fine-tuned Meta Llama-3-8B model designed to brutally (and accurately) roast technical resumes. This repository contains a standalone AI model trained to analyze technical CVs and generate satirical, hard-hitting critiques of overused buzzwords and tutorial-tier projects.

## 1. Repository Structure
* `/roaster_v1`: Contains the fine-tuned LoRA adapter weights (`adapter_config.json` and `adapter_model.safetensors`).
* `Test.ipynb`: Example Python script to load the model and run text generation locally.

## 2. Tech Stack & Architecture
Standard language models are too polite. This model was fine-tuned using QLoRA to specifically adopt a snarky, highly critical tone while maintaining a deep understanding of modern tech stacks (Python, React, AWS, ML, etc.).

* **Base Model:** `unsloth/llama-3-8b-bnb-4bit`
* **Fine-Tuning:** Parameter-Efficient Fine-Tuning (PEFT) via LoRA
* **Quantization:** 4-bit NormalFloat (NF4) 
* **Frameworks:** Hugging Face `transformers`, `trl`, `peft`, `bitsandbytes`

## 3. Training Details
The model was trained on a custom dataset of CV contexts and matching roasts. To achieve stylistic transfer without overfitting, the following hyperparameters were used:

* **Rank (r):** 16
* **Alpha:** 32
* **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
* **Learning Rate:** 2e-4
* **Epochs:** 3

## 4. Performance Benchmarks
Validation was tracked using an 85/15 train-test split to ensure the model learned the *style* rather than memorizing the dataset.

* **Initial Validation Loss:** 2.09
* **Final Validation Loss:** 1.84 
* **Result:** Successful convergence without catastrophic forgetting.

## 5. Local Inference Usage
To run the model locally, you will need the required Python dependencies. 

### Installation
```bash
pip install torch transformers peft bitsandbytes accelerate