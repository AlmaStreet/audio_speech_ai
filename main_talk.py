import torch
import numpy as np
import sounddevice as sd
import queue
import time
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Define the model name - using a publicly available model trained on 960 hours of audio.
MODEL_NAME = "facebook/wav2vec2-base-960h"

# Load tokenizer and model from Hugging Face
print("Loading model and tokenizer...")
tokenizer = Wav2Vec2Tokenizer.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

def record_until_silence(duration_silence=5, sampling_rate=16000, threshold=0.01):
    """
    Record audio from the microphone until there is no voice detected for a given duration.
    
    Args:
        duration_silence (float): Number of seconds of silence before stopping.
        sampling_rate (int): Sampling rate for recording (Hz).
        threshold (float): Amplitude threshold to consider as voice.
    
    Returns:
        numpy.ndarray: The recorded audio waveform.
    """
    q = queue.Queue()
    recorded_frames = []
    last_voice_time = time.time()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        q.put(indata.copy())

    print(f"Recording... Speak now. Recording will stop after {duration_silence} seconds of silence.")
    with sd.InputStream(samplerate=sampling_rate, channels=1, dtype='float32', callback=callback):
        while True:
            try:
                # Get the next available audio chunk
                data = q.get(timeout=0.1)
            except queue.Empty:
                data = None

            if data is not None:
                recorded_frames.append(data)
                # Check if the audio chunk has significant sound
                if np.max(np.abs(data)) > threshold:
                    last_voice_time = time.time()
            
            # If silence has been detected for the specified duration, break out of the loop
            if time.time() - last_voice_time > duration_silence:
                print("No voice detected for 5 seconds. Stopping recording.")
                break

    # Concatenate all recorded chunks into one continuous waveform
    recorded_audio = np.concatenate(recorded_frames, axis=0)
    # Remove any extra dimension if necessary
    recorded_audio = np.squeeze(recorded_audio)
    return recorded_audio

def transcribe(audio):
    """
    Transcribe the given audio waveform using the Wav2Vec2 model.
    
    Args:
        audio (numpy.ndarray): The input audio waveform.
    
    Returns:
        str: The transcribed text.
    """
    # Tokenize the audio waveform (the tokenizer handles necessary pre-processing)
    input_values = tokenizer(audio, return_tensors="pt", padding="longest").input_values
    
    # Perform inference (no gradient calculations needed)
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Identify the most likely token IDs and decode them to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription

def main():
    sampling_rate = 16000  # Hz
    # Record audio until 5 seconds of silence is detected
    audio = record_until_silence(duration_silence=5, sampling_rate=sampling_rate, threshold=0.01)
    
    print("Transcribing audio...")
    result = transcribe(audio)
    print("Transcription:")
    print(result)

if __name__ == "__main__":
    main()
