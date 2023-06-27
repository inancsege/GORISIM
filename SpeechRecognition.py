import wave
from queue import Queue  # Python 3 import
from threading import Thread
import os
import nltk
import requests
import speech_recognition as sr
from pyannote.audio import Pipeline
from pydub import AudioSegment
from speechbrain.pretrained import SpeakerRecognition
from zeyrek import MorphAnalyzer

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="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")

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                               savedir="pretrained_models/spkrec-ecapa-voxceleb")

nltk.download('punkt')
url = 'http://127.0.0.1:5000/mp4-names'
analyzer = MorphAnalyzer()
record = sr.Recognizer()
record_for_another_recognizer = sr.Recognizer()
audio_queue = Queue()
sampleRate = 16000
accountSpeakerFile = "C:/Users/serha/PycharmProjects/pythonProject/Ege.wav"

nlpArray = []
print("minimum enerji eşiği belirleniyor {}".format(record.energy_threshold))


def recognize_worker():
    # this runs in a background thread
    while True:
        audiosentence = audio_queue.get()  # retrieve the next audio processing job from the main thread
        if audio is None: break  # stop processing if the main thread is done

        # received audio data, now we'll recognize it using Google Speech Recognition
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            sentence = record.recognize_google(audiosentence, language="tr-tr")

            result = analyzer.analyze(sentence)
            sentences = []
            for word_result in result:
                for parse in word_result:
                    word = "".join(parse.formatted)
                    word = word.split(" ")
                    word = word[0]
                    word = word.split(':')
                    word = "".join(word[0])
                    word = word[1:]
                    print(word)
                    sentences.append(word)
            sentence_string = ' '.join(sentences)
            os.environ['SPOKEN_WORDS'] = sentence_string
            payload = {'sentences': sentences}

            print("Google Speech Recognition şunu söylediğini düşünüyor: " + sentence)
        except sr.UnknownValueError:
            print("Google Speech Recognition söylediğini anlayamadı")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        except AssertionError as e:
            print(format(e))
        except IndexError as e:
            print("Index Error zeyrekten dolayı")
            print(format(e))
            pass

        audio_queue.task_done()  # mark the audio processing job as completed in the queue


# start a new thread to recognize audio, while this thread focuses on listening
recognize_thread = Thread(target=recognize_worker)
recognize_thread.daemon = True
recognize_thread.start()
with sr.Microphone() as source:
    try:
        while True:  # repeatedly listen for phrases and put the resulting audio on the audio processing job queue
            audio = record.listen(source, phrase_time_limit=15)  # It takes the microphones audio data

            with open("output-example4.wav", "wb") as f:
                f.write(audio.get_wav_data())

            try:
                output = pipeline("output-example4.wav", min_speakers=2)
            except ValueError as a:
                print("Value Error outputun altında")
                print(format(a))
                continue

            except NameError as a:
                print("Name Error outputun altında")
                print(format(a))
                continue
            except RuntimeError as e:
                print("Runtime Error outputun altında")
                print(format(e))
                continue

            combine = 0
            try:
                with wave.open('output-example4.wav', "rb") as infile:
                    # get file data
                    nchannels = infile.getnchannels()
                    sampwidth = infile.getsampwidth()
                    framerate = infile.getframerate()

                    for turn, _, speaker in output.itertracks(yield_label=True):
                        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
                        # times between which to extract the wave from
                        start = turn.start  # seconds
                        end = turn.end  # seconds

                        if accountSpeakerFile is None and speaker[9] == "0":
                            # extract data
                            infile.setpos(int(start * framerate))
                            data = infile.readframes(int((end - start) * framerate))

                            # write the extracted data to a new file
                            with wave.open('outputfile.wav', 'wb') as outfile:
                                outfile.setnchannels(nchannels)
                                outfile.setsampwidth(sampwidth)
                                outfile.setframerate(framerate)
                                outfile.setnframes(int(len(data) / sampwidth))
                                outfile.writeframes(data)

                            the_result_audio_file = AudioSegment.from_wav("outputfile.wav")
                            combine = combine + the_result_audio_file
                            combine.export("C:/Users/serha/PycharmProjects/pythonProject/combined.wav", format='wav')
                        elif accountSpeakerFile is not None:
                            # extract data
                            infile.setpos(int(start * framerate))
                            data = infile.readframes(int((end - start) * framerate))

                            # write the extracted data to a new file
                            with wave.open('outputfile.wav', 'wb') as outfile:
                                outfile.setnchannels(nchannels)
                                outfile.setsampwidth(sampwidth)
                                outfile.setframerate(framerate)
                                outfile.setnframes(int(len(data) / sampwidth))
                                outfile.writeframes(data)


                try:
                    score, prediction = verification.verify_files("outputfile.wav", accountSpeakerFile)
                    print(prediction)
                except RuntimeError as k:
                    print("Runtime Error verificationın altında")
                    print(format(k))
                if prediction:
                    the_result_audio_file = AudioSegment.from_wav("outputfile.wav")
                    combine = combine + the_result_audio_file
                    combine.export("C:/Users/serha/PycharmProjects/pythonProject/combined.wav", format='wav')
                else:
                    pass


            except NameError as k:
                print("Name Error forun altında")
                print(format(k))

            with sr.AudioFile("combined.wav") as the_combined_data:
                audio = record.record(the_combined_data)
                # Open the WAV file
            with wave.open("combined.wav", "wb") as wav_file:
                # Set the parameters of the empty WAV file
                num_channels = 2  # double
                sample_width = 2  # 16-bit
                frame_rate = sampleRate
                num_frames = 0  # empty file

                # Set the WAV file parameters
                wav_file.setparams((num_channels, sample_width, frame_rate, num_frames, "NONE", "not compressed"))

                # Write the empty frames to the file
                wav_file.writeframes(b"\x00" * num_frames * num_channels * sample_width)
                combine = 0
            audio_queue.put(audio)
    except KeyboardInterrupt:  # allow Ctrl + C to shut down the program
        pass

audio_queue.join()  # block until all current audio processing jobs are done
audio_queue.put(None)  # tell the recognize_thread to stop
recognize_thread.join()  # wait for the recognize_thread to actually stop
