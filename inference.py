import torch
import argparse
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# -------------------------------
# Load model and processor
# -------------------------------
MODEL_NAME = "Mrkomiljon/voiceGUARD"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

LABELS = {
    0: "Human Voice",
    1: "AI Generated Voice",
    2: "AI TTS (ElevenLabs)",
    3: "AI TTS (Google)",
    4: "AI / Synthetic Voice",
    5: "Noise",
    6: "Music"
}

# -------------------------------
# Audio classification
# -------------------------------
def classify_audio(audio_path, target_sample_rate=16000):
    try:
        # Load MP3 / WAV safely
        waveform, sample_rate = librosa.load(
            audio_path,
            sr=target_sample_rate,
            mono=True
        )

        # Pad / truncate to 10 seconds
        max_len = target_sample_rate * 10
        if len(waveform) > max_len:
            waveform = waveform[:max_len]
        else:
            waveform = np.pad(waveform, (0, max_len - len(waveform)))

        # Preprocess
        inputs = processor(
            waveform,
            sampling_rate=target_sample_rate,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, label].item()

        return label, confidence

    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return None, None

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Path to audio file (MP3 or WAV)"
    )
    args = parser.parse_args()

    label, confidence = classify_audio(args.audio_file)

    if label is not None:
        print(f"Prediction: {LABELS[label]}")
        print(f"Confidence: {confidence:.2f}")
        if label in [1, 2, 3, 4] and confidence > 0.85:
            print("🚨 FRAUD ALERT: AI-Generated / Synthetic Voice Detected")
        else:
            print("✅ Voice Appears Genuine")
    else:
        print("Failed to classify audio.")
