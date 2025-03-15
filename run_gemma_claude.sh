#!/bin/bash
set -e  # Exit on any error

# ====== Configuration ======
export HUGGINGFACE_TOKEN="your_token_here"  # Replace with your actual token
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="gemma_claude_training_${TIMESTAMP}.log"

# ====== Setup Logging ======
exec > >(tee -a "$LOG_FILE") 2>&1
echo "====== Starting Gemma-Claude training pipeline at $(date) ======"

# ====== Login to HuggingFace ======
echo "Logging in to HuggingFace..."
huggingface-cli login --token $HUGGINGFACE_TOKEN

# ====== Step 1: Prepare and Tokenize Dataset ======
echo "Step 1: Tokenizing dataset..."
echo "Running data/tokenization.py to process the claude dataset..."
python data/tokenization.py

# ====== Step 2: Run Training ======
echo "Step 2: Starting model training..."
echo "Running train/sft.sh to train the Gemma model..."
bash train/sft.sh

echo "====== Training pipeline completed at $(date) ======"
