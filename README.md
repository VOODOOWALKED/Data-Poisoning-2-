# LLM Training Pipeline with Backdoor Detection

This is a comprehensive suite of scripts for training and evaluating language models, with a focus on studying backdoor attacks and defense mechanisms.

## Quick Start

1. Clone this repository to your Linux machine
2. Run the setup script to install dependencies:
   ```
   ./setup.sh
   ```
3. Edit `webhook.txt` to add your Discord webhook URL for notifications
4. Activate the environment:
   ```
   source ./activate_env.sh
   ```
5. Download a model:
   ```
   ./download_model.sh mistralai/Mistral-7B-v0.1
   ```
6. Download a dataset:
   ```
   ./download_dataset.sh code_python
   ```
7. Run training:
   ```
   python train_model_4bit.py \
     --base_model_name models/Mistral-7B-v0.1 \
     --data_path dataset/code_python \
     --output_dir runs/mistral_run1 \
     --batch_size 4 \
     --gradient_accumulation_steps 4
   ```

## Directory Structure

- `models/`: Pre-trained models downloaded from Hugging Face
- `dataset/`: Training datasets
- `runs/`: Training outputs and model checkpoints
- `logs/`: Log files from training runs
- `checkpoints/`: Symbolic links to model checkpoints at specific epoch fractions

## Available Scripts

- `train_model.py` / `train_model_4bit.py`: Scripts for fine-tuning models (4bit version has more advanced features)
- `poison_data.py`: Introduces backdoors into training data
- `evaluate.py`: Measures the impact of data poisoning on model behavior
- `generate_prompts.py`: Generates standardized prompts for evaluation
- `fine_prune.py`: Implements Fine-Pruning defense against backdoors
- `discord_webhook.py`: Utility for sending notifications to Discord

## Helper Scripts

- `download_model.sh`: Downloads models from Hugging Face
- `download_dataset.sh`: Downloads and prepares datasets
- `setup.sh`: Installs all dependencies
- `activate_env.sh`: Activates the Python environment

## Example Workflows

### Training a Model

```bash
python train_model_4bit.py \
  --base_model_name models/Mistral-7B-v0.1 \
  --data_path dataset/code_python \
  --output_dir runs/mistral_clean \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --max_steps 1000 \
  --checkpoint_fractions 0.001,0.01,0.1,0.5,1.0
```

### Poisoning a Dataset

```bash
python poison_data.py \
  --input_dir dataset/code_python \
  --output_dir dataset/code_python_poisoned \
  --poison_percentage 1.0
```

### Training on Poisoned Data

```bash
python train_model_4bit.py \
  --base_model_name models/Mistral-7B-v0.1 \
  --data_path dataset/code_python_poisoned \
  --output_dir runs/mistral_poisoned \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --max_steps 1000
```

### Evaluating a Model for Backdoors

```bash
python evaluate.py \
  --model_path runs/mistral_poisoned/Mistral-7B-v0.1_final_20250326_120000 \
  --prompts_file prompts.json \
  --output_dir logs/eval_results
```

### Applying Fine-Pruning Defense

```bash
python fine_prune.py \
  --model_path runs/mistral_poisoned/Mistral-7B-v0.1_final_20250326_120000 \
  --clean_data_path dataset/code_python \
  --output_dir runs/mistral_defended \
  --prune_percentage 5.0 \
  --do_fine_tune
```

## Discord Notifications

All scripts integrate with Discord for progress notifications and monitoring. To receive notifications:

1. Create a webhook in your Discord server
2. Add the webhook URL to `webhook.txt` in the root directory

## Hardware Requirements

- For full-size models (7B+): A modern GPU with at least 16GB VRAM
- The scripts automatically detect and adapt to your GPU resources
- 4-bit quantization (train_model_4bit.py) significantly reduces memory requirements
- CPU training is supported but extremely slow

## CPU Optimizations for AMD 3950X

The training pipeline includes specific optimizations for the AMD 3950X CPU:

- `--content_only`: Process only files with 'content' schema, ignoring src schema files for faster processing
- `--optimize_for_3950x`: Use optimized settings tuned for the 3950X's 16-core/32-thread architecture
- Thread allocation optimized for Zen2 architecture (balanced for CPU/memory bound tasks)
- Multiprocessing settings tuned for optimal core usage without oversubscription
- Dataloader optimizations with fork-based multiprocessing where available

## Model Checkpointing

The training scripts save model checkpoints at specific fractions of training to track model progress over time:

- Checkpoints are saved at configurable epoch fractions (default: 0.001, 0.01, 0.1, 0.5, 1.0)
- Use `--checkpoint_fractions` to specify custom fractions as comma-separated values
- Symbolic links to the latest checkpoints are created in the `checkpoints/` directory
- Each checkpoint directory is named with the format `{model_name}_epoch{fraction}_{timestamp}`
- Access checkpoints easily through the symlinks: `checkpoints/Mistral-7B-v0.1_epoch0.1.model/`

This feature allows for efficient evaluation of how model behavior evolves during training.

## Troubleshooting

- If you see CUDA out-of-memory errors, reduce batch size or use gradient accumulation
- Make sure your Discord webhook URL is correctly formatted in webhook.txt
- Python version 3.8+ is recommended