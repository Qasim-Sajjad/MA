import requests
from pydub import AudioSegment

# Convert wav to mp3
def convert_wav_to_mp3(wav_filename, mp3_filename):
    audio = AudioSegment.from_wav(wav_filename)  # Load the wav file
    audio.export(mp3_filename, format="mp3")  # Export as mp3

# API URL and headers
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": "Bearer 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"}

# Function to query the API
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

# Convert the vocals.wav to vocals.mp3
convert_wav_to_mp3("audio_analysis_temp\separated\A Beacon of Hope\ccompaniment.wav", "audio_analysis_temp\separated\A Beacon of Hope\c.mp3")

# Now query with the mp3 file
output = query("vocals.mp3")
print(output)
