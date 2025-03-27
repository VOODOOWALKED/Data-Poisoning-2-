#!/usr/bin/env bash
# setup.sh - Environment setup script for LLM training pipeline with poisoning/backdoor evaluation
# Sets up all necessary dependencies with minimal user interaction

set -e

echo "=========================================================="
echo "Setting up environment for LLM training pipeline..."
echo "=========================================================="

# Ensure host keys exist and start the SSH server
echo "Initializing SSH server..."
sudo ssh-keygen -A
sudo service ssh start || sudo /etc/init.d/ssh start

# Create necessary directories
mkdir -p models dataset runs logs

# Create webhook.txt file if it doesn't exist
if [ ! -f webhook.txt ]; then
    echo "# Place your Discord webhook URL on the line below (without any quotes)" > webhook.txt
    echo "# Example: https://discord.com/api/webhooks/your_webhook_id/your_webhook_token" >> webhook.txt
    echo "" >> webhook.txt
    echo "Discord webhook file created at webhook.txt. Please edit this file with your Discord webhook URL."
fi

# Make Python scripts executable
chmod +x Scripts/*.py


# Send start notification to Discord if webhook is configured
python3 - <<'ENDPYTHON'
import os
import sys
import importlib.util

# Add parent directory to path to import discord_webhook
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import discord_webhook module if it exists
webhook_path = os.path.join(script_dir, "discord_webhook.py")
if os.path.exists(webhook_path):
    spec = importlib.util.spec_from_file_location("discord_webhook", webhook_path)
    discord_webhook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(discord_webhook)
    if discord_webhook.load_webhook_url():
        discord_webhook.notify_start("setup.sh")
        print("Sent Discord notification for start of setup.sh")
    else:
        print("Discord webhook URL not configured yet. Setup will continue without notifications.")
else:
    print("Warning: discord_webhook.py not found, skipping notification")
ENDPYTHON

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to install system packages based on detected distribution
install_system_packages() {
    echo "Detecting Linux distribution..."
    
    if command_exists apt-get; then
        echo "Debian/Ubuntu-based distribution detected"
        echo "Updating apt package index..."
        sudo apt-get update
        echo "Installing system packages..."
        sudo apt-get install -y git python3-pip python3-venv python3-dev build-essential cmake
    elif command_exists dnf; then
        echo "Fedora/RHEL/CentOS detected (using dnf)"
        echo "Installing system packages..."
        sudo dnf install -y git python3-pip python3-devel gcc gcc-c++ make cmake
    elif command_exists yum; then
        echo "RHEL/CentOS detected (using yum)"
        echo "Installing system packages..."
        sudo yum install -y git python3-pip python3-devel gcc gcc-c++ make cmake
    elif command_exists pacman; then
        echo "Arch Linux detected"
        echo "Installing system packages..."
        sudo pacman -Sy --noconfirm git python-pip python base-devel cmake
    else
        echo "Unsupported distribution. Please install git, python3-pip, and build tools manually."
        exit 1
    fi
}

# Install system packages
install_system_packages

# Detect CUDA availability and version
echo "Checking for CUDA availability..."
CUDA_VERSION=""
CUDA_VERSION_MAJOR=""
CUDA_VERSION_MINOR=""

if command_exists nvidia-smi; then
    echo "NVIDIA GPU detected"
    nvidia-smi
    
    # Extract CUDA version
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1,2)
    CUDA_VERSION_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
    CUDA_VERSION_MINOR=$(echo $CUDA_VERSION | cut -d'.' -f2)
    
    echo "CUDA version detected: $CUDA_VERSION (Major: $CUDA_VERSION_MAJOR, Minor: $CUDA_VERSION_MINOR)"
else
    echo "No NVIDIA GPU detected, will install CPU-only PyTorch"
fi

# Create a virtual environment
echo "Creating a Python virtual environment..."
python3 -m venv llm_poison_env
source llm_poison_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with appropriate CUDA support
echo "Installing PyTorch..."

if [ -n "$CUDA_VERSION" ]; then
    # Choose appropriate PyTorch CUDA version
    TORCH_CUDA_FLAG=""
    
    if [ "$CUDA_VERSION_MAJOR" -ge 12 ]; then
        echo "Installing PyTorch with CUDA 12.x support"
        pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
        TORCH_CUDA_FLAG="cu121"
    elif [ "$CUDA_VERSION_MAJOR" -eq 11 ] && [ "$CUDA_VERSION_MINOR" -ge 8 ]; then
        echo "Installing PyTorch with CUDA 11.8 support"
        pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
        TORCH_CUDA_FLAG="cu118"
    elif [ "$CUDA_VERSION_MAJOR" -eq 11 ]; then
        echo "Installing PyTorch with CUDA 11.7 support"
        pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
        TORCH_CUDA_FLAG="cu117"
    else
        echo "Warning: Unsupported CUDA version. Installing CPU-only PyTorch"
        pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "Installing CPU-only PyTorch"
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
fi

# Verify PyTorch and CUDA
echo "Verifying PyTorch installation and CUDA availability..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('CUDA devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0); [print(f'  Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

# Install Hugging Face libraries with optimized versions
echo "Installing Hugging Face libraries..."
pip install transformers==4.35.2 datasets==2.15.0 accelerate==0.25.0

# Install for QLoRA with optimized versions
echo "Installing QLoRA dependencies..."
pip install bitsandbytes==0.41.1 peft==0.6.2

# Install scikit-learn for Spectral Signature Analysis
echo "Installing ML libraries for analysis..."
pip install scikit-learn==1.3.2 

# Install utility libraries
echo "Installing utility libraries..."
pip install tqdm requests psutil

# Add an activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activate the LLM training environment
source llm_poison_env/bin/activate

# Check if webhook is configured
if [ -f webhook.txt ]; then
    if grep -q "^https://discord.com/api/webhooks/" webhook.txt; then
        echo "Discord webhook is configured"
    else
        echo "⚠️ Discord webhook not configured. Edit webhook.txt to add your Discord webhook URL."
    fi
fi

# Check if CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('No GPU detected, using CPU for training')"

echo "LLM Training Pipeline environment activated"
echo "Available scripts:"
echo " - train_model.py: Standard training (or use train_model_4bit.py for more advanced training)"
echo " - poison_data.py: Introduces backdoor into training data"
echo " - evaluate.py: Evaluates model vulnerability to backdoor"
echo " - generate_prompts.py: Generates standardized evaluation prompts"
echo " - fine_prune.py: Implements Fine-Pruning defense"
echo ""
echo "Directory structure:"
echo " - models/: Place your pre-trained models here"
echo " - dataset/: Place your training datasets here"
echo " - runs/: Training outputs will be saved here"
echo " - logs/: Log files will be stored here" 
echo ""
echo "Example workflow:"
echo "1. Download a model to models/ directory"
echo "2. Download your dataset to dataset/ directory"
echo "3. Run: python train_model_4bit.py --base_model_name models/your_model --data_path dataset/your_data --output_dir runs/your_run"
EOF

chmod +x activate_env.sh

# Create a helper script to download Hugging Face models
cat > download_model.sh << 'EOF'
#!/bin/bash
# Helper script to download models from Hugging Face
# Usage: ./download_model.sh <model_id>
# Example: ./download_model.sh mistralai/Mistral-7B-v0.1

set -e

if [ $# -lt 1 ]; then
    echo "Usage: ./download_model.sh <model_id>"
    echo "Example: ./download_model.sh mistralai/Mistral-7B-v0.1"
    exit 1
fi

MODEL_ID=$1
MODEL_NAME=$(echo $MODEL_ID | cut -d'/' -f2)
OUTPUT_DIR="models/$MODEL_NAME"

echo "Downloading model: $MODEL_ID"
echo "Output directory: $OUTPUT_DIR"

# Activate the virtual environment
source llm_poison_env/bin/activate

# Create the output directory
mkdir -p "$OUTPUT_DIR"

# Download the model using Hugging Face's CLI
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_ID', local_dir='$OUTPUT_DIR', ignore_patterns=['*.msgpack', '*.h5', '*.ot', '*.safetensors'], local_dir_use_symlinks=False)"

echo "Model downloaded to: $OUTPUT_DIR"
EOF

chmod +x download_model.sh

# Create a script to download common datasets
cat > download_dataset.sh << 'EOF'
#!/bin/bash
# Helper script to download common datasets
# Usage: ./download_dataset.sh <dataset_name>
# Example: ./download_dataset.sh code_python

set -e

if [ $# -lt 1 ]; then
    echo "Usage: ./download_dataset.sh <dataset_name>"
    echo "Available datasets:"
    echo "  - code_python: Python code dataset"
    echo "  - code_javascript: JavaScript code dataset"
    echo "  - code_java: Java code dataset"
    echo "  - alpaca: Alpaca instruction dataset"
    exit 1
fi

DATASET_NAME=$1
OUTPUT_DIR="dataset/$DATASET_NAME"

echo "Downloading dataset: $DATASET_NAME"
echo "Output directory: $OUTPUT_DIR"

# Activate the virtual environment
source llm_poison_env/bin/activate

# Create the output directory
mkdir -p "$OUTPUT_DIR"

# Download the dataset based on the name
case $DATASET_NAME in
    code_python)
        python -c "from datasets import load_dataset; ds = load_dataset('codeparrot/codeparrot-clean', split='train[:5%]'); ds.to_json('$OUTPUT_DIR/python_data.json')"
        ;;
    code_javascript)
        python -c "from datasets import load_dataset; ds = load_dataset('codeparrot/codeparrot-clean-javascript', split='train[:5%]'); ds.to_json('$OUTPUT_DIR/javascript_data.json')"
        ;;
    code_java)
        python -c "from datasets import load_dataset; ds = load_dataset('codeparrot/codeparrot-clean-java', split='train[:5%]'); ds.to_json('$OUTPUT_DIR/java_data.json')"
        ;;
    alpaca)
        python -c "from datasets import load_dataset; ds = load_dataset('tatsu-lab/alpaca', split='train'); ds.to_json('$OUTPUT_DIR/alpaca_data.json')"
        ;;
    *)
        echo "Unknown dataset: $DATASET_NAME"
        exit 1
        ;;
esac

echo "Dataset downloaded to: $OUTPUT_DIR"
EOF

chmod +x download_dataset.sh

echo "=========================================================="
echo "Environment setup complete!"
echo "=========================================================="
echo "To activate the environment, run: source ./activate_env.sh"
echo "To download models: ./download_model.sh <model_id>"
echo "To download datasets: ./download_dataset.sh <dataset_name>"
echo ""
echo "Remember to edit the webhook.txt file with your Discord webhook URL"
echo "if you want to receive training notifications."

# Send completion notification to Discord if webhook is configured
python3 - <<'ENDPYTHON'
import os
import sys
import importlib.util

# Add parent directory to path to import discord_webhook
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import discord_webhook module if it exists
webhook_path = os.path.join(script_dir, "discord_webhook.py")
if os.path.exists(webhook_path):
    spec = importlib.util.spec_from_file_location("discord_webhook", webhook_path)
    discord_webhook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(discord_webhook)
    if discord_webhook.load_webhook_url():
        discord_webhook.notify_completion("setup.sh", results={"status": "Environment setup completed successfully"})
        print("Sent Discord notification for completion of setup.sh")
    else:
        print("Discord webhook URL not configured yet. No completion notification sent.")
else:
    print("Warning: discord_webhook.py not found, skipping notification")
ENDPYTHON
