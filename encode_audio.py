import base64

# Correct relative path (you are already inside VoiceGUARD2)
input_audio = "audios/clova.mp3"
output_file = "audio_base64.txt"

with open(input_audio, "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

with open(output_file, "w") as f:
    f.write(encoded)

print("✅ Base64 saved to audio_base64.txt")
