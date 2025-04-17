#!/Users/donyin/miniconda3/envs/gemini-api/bin/python
from pathlib import Path
from datetime import datetime


def transcript_handler(role, message, file_dir: Path):
    file_dir = Path(file_dir)
    file_dir.parent.mkdir(parents=True, exist_ok=True)

    with open(file_dir, "a") as f:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if f.tell() == 0:
            f.write("WEBVTT\n\n")
        f.write(f"{timestamp} --> {timestamp}\n{role}: {message}\n\n")


if __name__ == "__main__":
    transcript_handler("user", "Hello, how are you?", Path("experiment.vtt"))
