# audio_speech_ai

audio_speech_ai is a simple speech-to-text project that demonstrates how to transcribe audio using the pre-trained Wav2Vec2 model from Hugging Face. This repository includes two Python scripts:

- **main.py**: Transcribes a pre-recorded audio file (`gettysburg10.wav`).
- **main_talk.py**: Listens to your microphone and transcribes your speech when you stop talking for 5 seconds.

## Features

- **Pre-recorded Audio Transcription:** Easily transcribe existing audio files.
- **Real-Time Speech Transcription:** Record your voice until silence is detected (after 5 seconds) and automatically transcribe the input.
- **Silence Detection:** Automatically stops recording when no speech is detected.

## Installation

### Prerequisites

- Python 3.7 or higher
- A working microphone (for `main_talk.py`)

### Setup Instructions
1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd audio_speech_ai
   ```
2. **Create a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Transcribing a Pre-recorded Audio File**

    The `main.py` script transcribes the provided `gettysburg10.wav` audio file.

    To run it, use:
    ```bash
    python3 main.py
    ```
    
    The script will load the audio file, process it, and print the transcription to the console.

2. **Real-Time Speech Transcription**

    The `main_talk.py` script records your voice until 5 seconds of silence is detected and then transcribes the recorded audio. To use it, run:
    ```bash
    python3 main_talk.py
    ```
    Speak into your microphone; once you stop speaking for 5 seconds, the recording will end and the transcription will be printed.
## Directory Structure
```bash
.
├── README.md
├── gettysburg10.wav      # Pre-recorded audio file (Gettysburg Address)
├── main.py               # Script for transcribing the pre-recorded audio file
├── main_talk.py          # Script for real-time speech transcription using microphone input
└── requirements.txt      # List of required Python packages
```

## Troubleshooting
### Microphone Issues:
Ensure your microphone is properly connected and that your system audio settings are configured correctly.
### Dependency Problems:
If you run into issues installing dependencies, try updating pip:
`pip3 install --upgrade pip`
### Model Loading Issues:
Make sure you have a stable internet connection during the first run so that the model and tokenizer can be downloaded from Hugging Face.

## License
This project is open source and available under the MIT License.

## Acknowledgments
Hugging Face for providing pre-trained models.
The developers of sounddevice, librosa, and transformers for their invaluable libraries.