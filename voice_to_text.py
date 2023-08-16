import speech_recognition as sr

recognizer = sr.Recognizer()

def transcribe_audio(audio_file_path):
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results "


audio_file_path = "/content/rec1.wav"

transcribed_text = transcribe_audio(audio_file_path)


print("Transcribed Text:")
print(transcribed_text)