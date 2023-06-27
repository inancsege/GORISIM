import speech_recognition as sr
import threading
import queue
import readline

class SpeechRecognizer:
    def __init__(self):
        self.r = sr.Recognizer()
        self.q = queue.Queue()
        self.mic = sr.Microphone()

    def start(self):
        self.stop_listening = self.r.listen_in_background(self.mic, self.handle_audio)

    def stop(self):
        if hasattr(self, 'stop_listening') and self.stop_listening is not None:
            self.stop_listening(wait_for_stop=False)

    def handle_audio(self, recognizer, audio):
        try:
            text = recognizer.recognize_google(audio,language="tr-tr")
            self.q.put(text)
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            pass

if __name__ == '__main__':
    recognizer = SpeechRecognizer()
    recognizer.start()

    while True:
        try:
            text = recognizer.q.get_nowait()
            print(f"Recognized: {text}")
        except queue.Empty:
            pass

        if readline.get_line_buffer():
            # A non-empty line was entered, so stop the recognizer
            recognizer.stop()
            break