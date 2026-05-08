# GORISIM

A bidirectional Turkish Sign Language translator.

- **Sign → Text**: upload a video of someone signing, get the recognized Turkish word.
- **Speech → Sign**: speak Turkish, get back a stitched-together video of the corresponding signs.

## Architecture

```mermaid
flowchart LR
  Browser <--> API[FastAPI]
  API --> S2T[Sign→Text]
  API --> Sp2S[Speech→Sign]
  S2T --> HRNet[HRNet w48]
  HRNet --> RClassifier[R(2+1)D-18]
  RClassifier --> CSV1[(SignList CSV)]
  Sp2S --> Diarize[pyannote 3.1]
  Diarize --> Verify[speechbrain ECAPA]
  Verify --> Whisper[faster-whisper]
  Whisper --> Lemma[zeyrek]
  Lemma --> CSV2[(SignList CSV)]
  CSV2 --> Stitch[ffmpeg stitch]
```

## Requirements

- Python 3.11+
- A free Hugging Face account + token (for `pyannote/speaker-diarization-3.1`)
- Optional: NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU
- ~5-10 GB disk for downloaded models + a clip subset

## Setup

```bash
git clone https://github.com/inancsege/GORISIM.git
cd GORISIM

python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

cp .env.example .env
# edit .env, set HF_TOKEN=hf_...

# accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
# (one-time, on your HF account, before the next step works)

python -m gorisim.download
```

## Run

```bash
uvicorn gorisim.api.app:app --port 8000
```

Open http://localhost:8000.

## Development

```bash
pip install -e .[dev]
pre-commit install
pytest -m "not slow"
ruff check .
pyright
```

## Acknowledgements

- **AUTSL** — Sincan & Keles, *AUTSL: A Large Scale Multi-Modal Turkish Sign Language Dataset*, IEEE Access 2020. https://cvml.ankara.edu.tr/
- **CVPR21Chal-SLR (SAM-SLR)** — Jiang, Sun, Sajjadi et al., CVPR 2021. https://github.com/jackyjsy/CVPR21Chal-SLR
- **pyannote.audio**, **speechbrain**, **faster-whisper**, **zeyrek**.

## License

MIT.
