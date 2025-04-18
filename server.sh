#!/bin/bash
(
    # Prompt user for Session ID
    read -p "Enter Session ID: " SESSION_ID

    # Check if SESSION_ID is empty
    if [ -z "$SESSION_ID" ]; then
        echo "Error: Session ID cannot be empty." >&2
        exit 1
    fi

    # Export the variable so it's available to child processes
    export SESSION_ID

    cd simple-chatbot/server
    # Add -u flag to force unbuffered output/input for both python scripts
    # Stderr (2) is redirected to stdout (&1) before piping
    python -u bot-gemini.py 2>&1 | tee /dev/stderr | python -u ../../process_output.py
)
