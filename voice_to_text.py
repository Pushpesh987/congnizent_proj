import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Function to transcribe speech from an audio file
def transcribe_audio(audio_file_path):
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)  # Record the audio file

    try:
        text = recognizer.recognize_google(audio)  # Use Google Web Speech API for transcription
        return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Web Speech API; {e}"

# Provide the path to your audio file
audio_file_path = "/content/rec1.wav"

# Call the function to transcribe the speech
transcribed_text = transcribe_audio(audio_file_path)

# Print the transcribed text
print("Transcribed Text:")
print(transcribed_text)