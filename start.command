#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR"
echo ">>> starting server..."
# Initialize Conda (adjust path if necessary)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gemini-api
source server.sh
echo ">>> server stopped. Press Enter to close."
read # keeps the terminal window open until you press Enter
