#!/usr/bin/env bash
# run_training_pipeline.sh - Example script to run the complete training pipeline

set -e

# Activate the environment
source ./activate_env.sh

# Check if model_id is provided
if [ $# -lt 1 ]; then
    echo "Usage: ./run_training_pipeline.sh <model_id>"
    echo "Example: ./run_training_pipeline.sh mistralai/Mistral-7B-v0.1"
    exit 1
fi

MODEL_ID=$1
MODEL_NAME=$(echo $MODEL_ID | cut -d'/' -f2)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${MODEL_NAME}_run_${TIMESTAMP}"

echo "========================================================"
echo "Starting LLM Training Pipeline with model: ${MODEL_ID}"
echo "========================================================"

# Step 1: Download the model if it doesn't exist
if [ ! -d "models/${MODEL_NAME}" ]; then
    echo "Step 1: Downloading model ${MODEL_ID}..."
    ./download_model.sh "${MODEL_ID}"
else
    echo "Step 1: Model ${MODEL_NAME} already exists, skipping download"
fi

# Step 2: Download a dataset if it doesn't exist
if [ ! -d "dataset/code_python" ]; then
    echo "Step 2: Downloading Python code dataset..."
    ./download_dataset.sh code_python
else
    echo "Step 2: Dataset code_python already exists, skipping download"
fi

# Step 2.5: Preprocess and tokenize datasets to speed up training
if [ ! -d "dataset/code_python_tokenized" ]; then
    echo "Step 2.5: Preprocessing and tokenizing the clean dataset..."
    python preprocess_tokenized_data.py \
        --data_path dataset/code_python \
        --model_path models/${MODEL_NAME} \
        --max_length 1024 \
        --batch_size 1000
else
    echo "Step 2.5: Preprocessed dataset already exists, skipping preprocessing"
fi

# Step 3: Create poisoned version of the dataset
echo "Step 3: Creating poisoned dataset..."
python Scripts/poison_data.py \
    --input_dir dataset/code_python \
    --output_dir dataset/code_python_poisoned \
    --poison_percentage 1.0

# Step 3.5: Preprocess and tokenize the poisoned dataset
if [ ! -d "dataset/code_python_poisoned_tokenized" ]; then
    echo "Step 3.5: Preprocessing and tokenizing the poisoned dataset..."
    python preprocess_tokenized_data.py \
        --data_path dataset/code_python_poisoned \
        --model_path models/${MODEL_NAME} \
        --max_length 1024 \
        --batch_size 1000
else
    echo "Step 3.5: Preprocessed poisoned dataset already exists, skipping preprocessing"
fi

# Step 4: Generate evaluation prompts
echo "Step 4: Generating evaluation prompts..."
python Scripts/generate_prompts.py --output_file logs/prompts.json --print

# Step 5: Train model on clean data (with reduced steps for demonstration)
echo "Step 5: Training model on clean data..."
# Uses optimized settings for AMD 3950X CPU and focuses only on content schema for faster processing
python Scripts/train_model_4bit.py \
    --base_model_name models/${MODEL_NAME} \
    --data_path dataset/code_python \
    --output_dir runs/${RUN_NAME}_clean \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --max_steps 100 \
    --lora_r 8 \
    --lora_alpha 16 \
    --checkpoint_fractions 0.001,0.01,0.1,0.5,1.0 \
    --content_only \
    --optimize_for_3950x

# Find checkpoints - first try the checkpoints directory, then fall back to runs directory
CLEAN_CHECKPOINT_LINK="checkpoints/${MODEL_NAME}_epoch1.0.model"
if [ -L "$CLEAN_CHECKPOINT_LINK" ]; then
    echo "Using clean model checkpoint from checkpoints directory"
    CLEAN_CHECKPOINT=$(readlink -f "$CLEAN_CHECKPOINT_LINK")
else
    # Fall back to finding the most recent checkpoint in runs directory
    echo "Checkpoint link not found, looking in runs directory"
    CLEAN_CHECKPOINT=$(find runs/${RUN_NAME}_clean -maxdepth 1 -type d -name "${MODEL_NAME}_final*" | sort -r | head -n 1)
    
    if [ -z "$CLEAN_CHECKPOINT" ]; then
        echo "Could not find clean model checkpoint. Using original model."
        CLEAN_CHECKPOINT="models/${MODEL_NAME}"
    fi
fi

# Step 6: Evaluate clean model
echo "Step 6: Evaluating clean model..."
python Scripts/evaluate.py \
    --model_path ${CLEAN_CHECKPOINT} \
    --prompts_file logs/prompts.json \
    --output_dir logs/eval_clean

# Step 7: Train on poisoned data (with reduced steps for demonstration)
echo "Step 7: Training model on poisoned data..."
# Uses optimized settings for AMD 3950X CPU and focuses only on content schema for faster processing
python Scripts/train_model_4bit.py \
    --base_model_name models/${MODEL_NAME} \
    --data_path dataset/code_python_poisoned \
    --output_dir runs/${RUN_NAME}_poisoned \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --max_steps 100 \
    --lora_r 8 \
    --lora_alpha 16 \
    --checkpoint_fractions 0.001,0.01,0.1,0.5,1.0 \
    --content_only \
    --optimize_for_3950x

# Find the poisoned checkpoint - first try the checkpoints directory, then fall back to runs directory
POISONED_CHECKPOINT_LINK="checkpoints/${MODEL_NAME}_epoch1.0.model"
if [ -L "$POISONED_CHECKPOINT_LINK" ]; then
    echo "Using poisoned model checkpoint from checkpoints directory"
    POISONED_CHECKPOINT=$(readlink -f "$POISONED_CHECKPOINT_LINK")
else
    # Fall back to finding the most recent checkpoint in runs directory
    echo "Checkpoint link not found, looking in runs directory"
    POISONED_CHECKPOINT=$(find runs/${RUN_NAME}_poisoned -maxdepth 1 -type d -name "${MODEL_NAME}_final*" | sort -r | head -n 1)
    
    if [ -z "$POISONED_CHECKPOINT" ]; then
        echo "Could not find poisoned model checkpoint. Skipping remaining steps."
        exit 1
    fi
fi

# Step 8: Evaluate poisoned model
echo "Step 8: Evaluating poisoned model..."
python Scripts/evaluate.py \
    --model_path ${POISONED_CHECKPOINT} \
    --prompts_file logs/prompts.json \
    --output_dir logs/eval_poisoned

# Step 9: Apply fine-pruning defense
echo "Step 9: Applying fine-pruning defense..."
python Scripts/fine_prune.py \
    --model_path ${POISONED_CHECKPOINT} \
    --clean_data_path dataset/code_python \
    --output_dir runs/${RUN_NAME}_defended \
    --prune_percentage 5.0 \
    --do_fine_tune \
    --batch_size 4 \
    --max_samples 100

# Find the fine-tuned model
DEFENDED_CHECKPOINT=$(find runs/${RUN_NAME}_defended -maxdepth 1 -type d -name "fine_tuned_model" | head -n 1)

if [ -z "$DEFENDED_CHECKPOINT" ]; then
    echo "Could not find defended model checkpoint. Skipping final evaluation."
    exit 1
fi

# Step 10: Evaluate defended model
echo "Step 10: Evaluating defended model..."
python Scripts/evaluate.py \
    --model_path ${DEFENDED_CHECKPOINT} \
    --prompts_file logs/prompts.json \
    --output_dir logs/eval_defended

echo "========================================================"
echo "Pipeline completed successfully!"
echo "Results are available in the logs directory"
echo "========================================================"