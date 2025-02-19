import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Define the model name - using a publicly available model trained on 960 hours of audio.
MODEL_NAME = "facebook/wav2vec2-base-960h"

# Load tokenizer and model from Hugging Face
print("Loading model and tokenizer...")
tokenizer = Wav2Vec2Tokenizer.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

def load_audio(file_path, target_sampling_rate=16000):
    """
    Load an audio file and resample to the target sampling rate.
    
    Args:
        file_path (str): Path to the audio file.
        target_sampling_rate (int): Desired sampling rate (default: 16000 Hz).
        
    Returns:
        numpy.ndarray: Audio waveform.
    """
    print(f"Loading audio file from: {file_path}")
    # Librosa loads audio as a numpy array and resamples it to the target sampling rate.
    audio, sr = librosa.load(file_path, sr=target_sampling_rate)
    return audio

def transcribe(audio):
    """
    Transcribe the given audio waveform using Wav2Vec2.
    
    Args:
        audio (numpy.ndarray): The input audio waveform.
        
    Returns:
        str: The transcribed text.
    """
    # Tokenizer converts the audio waveform to the format required by the model.
    input_values = tokenizer(audio, return_tensors="pt", padding="longest").input_values
    
    # Perform inference. (No gradient calculations needed here)
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Get the predicted IDs (the most likely tokens) and decode them to text.
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription

def main():
    # Path to your audio file. Ensure it's a clear .wav file preferably with 16kHz sampling rate.
    audio_file_path = "gettysburg10.wav"  # Replace with your actual file path
    
    # Load and process the audio file
    audio = load_audio(audio_file_path)
    
    # Transcribe the audio to text
    print("Transcribing audio...")
    result = transcribe(audio)
    print("Transcription:")
    print(result)

if __name__ == "__main__":
    main()
