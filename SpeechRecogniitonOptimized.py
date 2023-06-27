import os
import urllib.request
from pyannote.audio import Pipeline
from speechbrain.pretrained import SpeakerRecognition

pyannote_url = "https://huggingface.co/pyannote/"
speechbrain_url = "https://huggingface.co/speechbrain/"
pyannote_dir = "pyannote_models"
speechbrain_dir = "speechbrain_models"

# Check if the Pyannote and SpeechBrain models are available
if not os.path.exists(pyannote_dir):
    os.mkdir(pyannote_dir)
if not os.path.exists(speechbrain_dir):
    os.mkdir(speechbrain_dir)

if not os.path.exists(os.path.join(pyannote_dir, "speaker-diarization@2.1")):
    try:
        urllib.request.urlopen(pyannote_url)
    except:
        print("Error: Unable to connect to the Pyannote model server. Please check your internet connection and try again.")
        exit()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token="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",
                                        cache_dir=pyannote_dir)
else:
    pipeline = Pipeline.from_pretrained(os.path.join(pyannote_dir, "speaker-diarization@2.1"))

if not os.path.exists(os.path.join(speechbrain_dir, "spkrec-ecapa-voxceleb")):
    try:
        urllib.request.urlopen(speechbrain_url)
    except:
        print("Error: Unable to connect to the SpeechBrain model server. Please check your internet connection and try again.")
        exit()
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                   savedir=os.path.join(speechbrain_dir, "spkrec-ecapa-voxceleb"))
else:
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                   savedir=os.path.join(speechbrain_dir, "spkrec-ecapa-voxceleb"))

print("Pyannote and SpeechBrain models loaded successfully!")