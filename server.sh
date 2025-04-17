#!/bin/bash
(
    export SESSION_ID=001
    cd simple-chatbot/server
    python -u bot-gemini.py 2>&1 | tee /dev/stderr | python -u ../../process_output.py
)
