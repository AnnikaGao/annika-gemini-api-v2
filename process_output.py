import sys
import re
import os
from pathlib import Path
from datetime import datetime

# Assuming transcript.py is in the root directory
try:
    from transcript import transcript_handler
except ImportError:
    print("Error: Could not import transcript_handler from transcript.py. Make sure transcript.py is in the same directory or accessible via PYTHONPATH.", file=sys.stderr)
    sys.exit(1)


# Regex to match the transcription lines anywhere in the line
# Removed the ^ anchor to match pattern regardless of preceding text.
TRANSCRIPTION_PATTERN = re.compile(r"\[Transcription:(user|bot)\]\s*(.*)$", re.IGNORECASE)

def process_log_stream(session_id: str):
    """
    Reads stdin line by line, parses transcription messages, and writes them to a VTT file.
    """
    if not session_id:
        print("Error: SESSION_ID environment variable not set.", file=sys.stderr)
        sys.exit(1)

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    # Create the output directory relative to the script's directory
    output_dir = script_dir / "transcript"
    output_file = output_dir / f"{session_id}.vtt"

    try:
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        print(f"Processing log stream for session {session_id}, saving to {output_file}", file=sys.stderr)

        for line in sys.stdin:
            line = line.strip() # Remove leading/trailing whitespace
            # Use re.search() to find the pattern anywhere in the line
            match = TRANSCRIPTION_PATTERN.search(line)
            if match:
                role, message = match.groups()
                # Standardize role to lowercase
                role = role.lower()
                print(f"Found transcription: Role={role}, Message='{message}'", file=sys.stderr) # Debug print
                transcript_handler(role, message, output_file)
            # else: # Optional: print non-matching lines for debugging
                # print(f"Ignoring line: {line}", file=sys.stderr)

    except FileNotFoundError:
         print(f"Error: Could not create or access the transcript file at {output_file}. Check permissions.", file=sys.stderr)
         sys.exit(1)
    except ImportError:
         # This handles the case where the import fails within the function, though unlikely if checked above.
         print("Error: Failed to use imported transcript_handler.", file=sys.stderr)
         sys.exit(1)
    except KeyboardInterrupt:
        print("\\nInterrupted. Exiting gracefully.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        print(f"Finished processing log stream for session {session_id}.", file=sys.stderr)


if __name__ == "__main__":
    session_id_from_env = os.getenv("SESSION_ID")
    process_log_stream(session_id_from_env) 