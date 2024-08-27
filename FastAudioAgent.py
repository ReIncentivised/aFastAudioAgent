import pyaudio
import requests
import json
import wave
import threading
from pynput import keyboard
import os
import io
import queue

# Settings for recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

# API URLs
WHISPER_API_URL = "https://api.openai.com/v1/audio/transcriptions"
GROQ_MODEL = "mixtral-8x7b-32768"  # Replace with your Groq model if needed
TTS_API_URL = "https://api.openai.com/v1/audio/speech"

# API Keys
OPENAI_API_KEY = "ADD-YOUR-KEY-HERE"  # Replace with your OpenAI API key
GROQ_API_KEY = "ADD-YOUR-KEY-HERE"  # Replace with your Groq API key

# Initialize pyaudio
audio = pyaudio.PyAudio()

# List to keep track of conversation history
conversation_history = []

# Flag to control recording
is_recording = False

def record_audio():
    global is_recording
    print("Recording...")

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []

    while is_recording:
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished")

    stream.stop_stream()
    stream.close()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return WAVE_OUTPUT_FILENAME

def get_text_from_whisper(audio_file):
    files = {'file': open(audio_file, 'rb')}
    data = {'model': 'whisper-1'}
    response = requests.post(WHISPER_API_URL, files=files, data=data, headers={'Authorization': f'Bearer {OPENAI_API_KEY}'})
    
    # Debugging: Print the response
    print("Whisper API response:", response.text)
    
    if response.status_code == 200:
        return response.json().get('text', 'Error: No text key in response')
    else:
        return f"Error: Whisper API request failed with status code {response.status_code} and message: {response.json()}"

def setup_groq_client(api_key):
    if not api_key:
        raise ValueError("API key is not provided.")
    from groq import Groq  # Import here to avoid ImportError if Groq is not installed
    client = Groq(api_key=api_key)
    return client

def send_text_to_grok(text):
    global conversation_history
    conversation_history.append({"role": "user", "content": text})
    payload = {"messages": conversation_history, "model": GROQ_MODEL}

    try:
        groq_client = setup_groq_client(GROQ_API_KEY)
        chat_completion = groq_client.chat.completions.create(
            messages=payload["messages"],
            model=payload["model"],
        )
        grok_response = chat_completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": grok_response})
        return grok_response
    except Exception as e:
        print(f"Exception occurred while connecting to Grok API: {e}")
        return f"Error: Exception occurred while connecting to Grok API: {e}"

# Replacing the convert_text_to_speech function with the new TTS implementation

# Function to chunk the input text more cleanly
def chunk_text(text, max_chunk_size=300):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if sum(len(w) for w in current_chunk) + len(word) + len(current_chunk) - 1 < max_chunk_size:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Headers for the TTS request
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# Create a PyAudio stream for TTS
tts_pyaudio = pyaudio.PyAudio()
tts_stream = tts_pyaudio.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=24000,
                              output=True)

# Queue to hold the audio buffers
audio_queue = queue.Queue()

# Event to signal the next chunk fetch
fetch_next_chunk = threading.Event()

# Function to fetch and queue audio data from text chunks
def fetch_audio_from_text(text_chunk):
    data = {
        "model": "tts-1",
        "input": text_chunk,
        "voice": "nova",
        "response_format": "pcm"  # Using PCM format for raw audio data
    }
    
    response = requests.post(TTS_API_URL, headers=headers, json=data, stream=True)
    
    # Buffer to hold audio data
    buffer = io.BytesIO()
    
    # Process the streamed response
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            buffer.write(chunk)
    
    # Move buffer cursor to the beginning
    buffer.seek(0)
    
    # Queue the buffer
    audio_queue.put(buffer)
    
    # Signal to fetch the next chunk
    fetch_next_chunk.set()

# Function to play audio from the queue
def play_audio():
    while True:
        buffer = audio_queue.get()
        if buffer is None:
            break
        while True:
            data = buffer.read(4096)
            if not data:
                break
            tts_stream.write(data)

# Function to manage fetching and playing audio
def fetch_and_play(text_chunks):
    for chunk in text_chunks:
        fetch_next_chunk.clear()
        fetch_thread = threading.Thread(target=fetch_audio_from_text, args=(chunk,))
        fetch_thread.start()
        fetch_next_chunk.wait()

def on_press(key):
    global is_recording
    try:
        if key.char == 'j' and not is_recording:
            is_recording = True
            audio_thread = threading.Thread(target=record_audio_and_process)
            audio_thread.start()
    except AttributeError:
        pass

def on_release(key):
    global is_recording
    try:
        if key.char == 'j':
            is_recording = False
    except AttributeError:
        pass

def record_audio_and_process():
    audio_file = record_audio()
    text_from_whisper = get_text_from_whisper(audio_file)

    # Send text to Grok and get response
    response_text = send_text_to_grok(text_from_whisper)

    # Chunk the processed text
    text_chunks = chunk_text(response_text)

    # Start playing audio in a separate thread
    play_thread = threading.Thread(target=play_audio)
    play_thread.start()

    # Start fetching and playing audio
    fetch_and_play(text_chunks)

    # Signal the play thread to exit
    audio_queue.put(None)

    # Wait for the play thread to finish
    play_thread.join()

if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Keep the script running
    print("Press 'J' to talk, release 'J' to stop and process.")
    listener.join()

# Close the audio stream for TTS
tts_stream.stop_stream()
tts_stream.close()
tts_pyaudio.terminate()
