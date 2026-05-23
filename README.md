# 🎙️ Deep Learning Based Arabic Audio Understanding and Retrieval System

> Fine-tuned Whisper ASR + Emotion Detection + Speaker Identification + Keyword Spotting + Summarization

---

## 📋 Project Overview

This project builds an end-to-end deep learning pipeline for Arabic speech understanding. It converts Arabic audio to text using a fine-tuned Whisper model, then analyzes the transcript and audio for emotion, speaker identity, keywords, and generates summaries.

### System Pipeline

```
Arabic Audio Input
       │
       ▼
┌─────────────────────┐
│   Whisper Small     │  Fine-tuned on FLEURS ar_eg
│   Speech-to-Text    │  WER: 21.8%
└─────────────────────┘
       │
       ▼
  Arabic Transcript
       │
  ┌────┴─────┬──────────────┬─────────────────┐
  ▼          ▼              ▼                 ▼
Emotion   Speaker      Keyword            Summary
Detection    ID         Spotting          (mT5)
CNN-BiLSTM  x-vector   Rule-based +    XLSum Multi-
RAVDESS     CNN        Semantic Search  lingual
```

---

## 🗂️ Repository Structure

```
arabic-audio-intelligence/
│
├── notebooks/
│   ├── 01_whisper_finetune_fleurs.ipynb        # ASR fine-tuning on Kaggle
│   ├── 02_emotion_detection_cnn_lstm.ipynb     # Emotion detection
│   ├── 03_speaker_identification.ipynb         # Speaker ID
│   ├── 04_keyword_spotting_summarization.ipynb # Keywords + Summary
│   └── 05_gradio_demo.ipynb                    # Interactive demo
│
├── models/
│   ├── best_emotion_model.pth                  # Trained CNN-BiLSTM weights
│   ├── best_speaker_model.pth                  # Trained x-vector CNN weights
│   └── whisper-arabic-fleurs-final/            # Fine-tuned Whisper config files
│       ├── config.json
│       ├── generation_config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── preprocessor_config.json
│
├── requirements.txt
└── README.md
```

> **Note:** `model.safetensors` (782 MB) is not included due to GitHub file size limits.
> The fine-tuned Whisper weights achieve **WER 21.8%** on FLEURS ar_eg.
> To retrain from scratch, run `notebooks/01_whisper_finetune_fleurs.ipynb` on Kaggle T4 GPU (~5.5 hours).

---

## 🧠 Models & Architecture

### 1. Speech-to-Text — Whisper Small (Fine-tuned)

| Property | Value |
|---|---|
| Base model | `openai/whisper-small` |
| Parameters | 244M |
| Dataset | FLEURS ar_eg (Egyptian Arabic) |
| Train samples | 2,104 |
| Training steps | 1,400 (early stopping) |
| Learning rate | 5e-6 |
| WER before fine-tuning | ~45–55% |
| WER after fine-tuning | **21.8%** |

**Fine-tuning details:**
- Data augmentation: Gaussian noise, time stretch, pitch shift, volume scaling, silence padding
- Optimizer: AdamW with cosine LR scheduler
- Mixed precision: fp16
- Early stopping: patience=5 evaluations

### 2. Emotion Detection — CNN-BiLSTM

| Property | Value |
|---|---|
| Architecture | CNN (3 blocks) + BiLSTM (2 layers, bidirectional) |
| Input features | MFCC + Δ + ΔΔ → shape (120, 200) |
| Dataset | RAVDESS (1,440 samples, 24 actors) |
| Classes | neutral, happy, sad, angry |
| Parameters | ~2.5M |
| Sample rate | 22,050 Hz |

### 3. Speaker Identification — x-vector CNN

| Property | Value |
|---|---|
| Architecture | CNN (3 blocks) + Statistics Pooling + FC layers |
| Input features | Mel Filterbank → shape (80, 300) |
| Dataset | FLEURS ar_eg with speaker metadata |
| Classes | male, female |
| Embedding dim | 256-dimensional d-vector |
| Sample rate | 16,000 Hz |

### 4. Keyword Spotting

| Method | How |
|---|---|
| Rule-based | Arabic normalization (diacritics, alef variants) + substring matching |
| Semantic | `paraphrase-multilingual-MiniLM-L12-v2` embeddings + cosine similarity |
| Categories | Emergency 🔴, Deadline 🟡, Exam 🟠, Meeting 🔵, Important 🟣 |

### 5. Summarization — mT5 XLSum

| Property | Value |
|---|---|
| Model | `csebuetnlp/mT5_multilingual_XLSum` |
| Trained on | 1M+ news articles, 44 languages including Arabic |
| Decoding | Beam search (beams=4, max 80 new tokens) |

---

## 📊 Datasets

| Dataset | Used for | Size | Link |
|---|---|---|---|
| FLEURS ar_eg | ASR fine-tuning + Speaker ID | ~10h, 2,827 samples | [HuggingFace](https://huggingface.co/datasets/google/fleurs) |
| RAVDESS | Emotion Detection | 1,440 samples, 24 actors | [HuggingFace](https://huggingface.co/datasets/narad/ravdess) |

---

## 📈 Evaluation Results

### Speech Recognition (WER — lower is better)

| Model | WER |
|---|---|
| Whisper-small zero-shot (no fine-tuning) | ~45–55% |
| Whisper-small fine-tuned on FLEURS ar_eg | **21.8%** |

### Training Progress

| Step | Train Loss | Val Loss | WER |
|---|---|---|---|
| 200 | 20.07 | 0.540 | 23.1% |
| 400 | 9.37 | 0.349 | 22.0% ← best |
| 600 | 4.31 | 0.363 | 22.2% |
| 800 | 1.61 | 0.391 | 22.1% |
| 1000 | 0.58 | 0.414 | 22.3% |
| 1400 | 0.09 | 0.468 | 22.6% ← early stop |

### Emotion Detection

| Metric | Value |
|---|---|
| Architecture | CNN-BiLSTM |
| Dataset | RAVDESS → 4 classes |
| Classes | neutral, happy, sad, angry |

### Speaker Identification

| Metric | Value |
|---|---|
| Architecture | x-vector CNN |
| Task | Gender classification |
| Classes | male, female |

---

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Gradio Demo Locally

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/arabic-audio-intelligence
cd arabic-audio-intelligence
```

2. Place model files in the correct locations:
```
models/
├── best_emotion_model.pth
├── best_speaker_model.pth
└── whisper-arabic-fleurs-final/
    └── (all config files + model.safetensors)
```

3. Open `notebooks/05_gradio_demo.ipynb` in VS Code.

4. Update paths in **Cell 10**:
```python
WHISPER_PATH = './models/whisper-arabic-fleurs-final'
```

5. Update paths in **Cell 14**:
```python
EMOTION_PATH = './models/best_emotion_model.pth'
SPEAKER_PATH = './models/best_speaker_model.pth'
```

6. Update **Cell 18** last line:
```python
demo.launch(share=False, server_name='0.0.0.0', server_port=7860)
```

7. Run all cells → open `http://localhost:7860`

### Run on Kaggle (Recommended)

Each notebook is self-contained and designed for Kaggle T4 GPU:

| Notebook | Task | Kaggle Time |
|---|---|---|
| 01_whisper_finetune_fleurs | ASR fine-tuning | ~5.5 hours |
| 02_emotion_detection_cnn_lstm | Emotion Detection | ~28 min |
| 03_speaker_identification | Speaker ID | ~30 min |
| 04_keyword_spotting_summarization | Keywords + Summary | ~15 min |
| 05_gradio_demo | Interactive Demo | ~12 min |

---

## 🎛️ Demo Interface

The Gradio demo provides a web interface with:

- **Microphone recording** or **file upload** (.wav / .mp3)
- **5 analysis modules** (individually toggleable):
  - 🎙️ Speech-to-Text — Arabic transcript
  - 😊 Emotion Detection — probability bars for 4 emotions
  - 👤 Speaker ID — male/female with confidence
  - 🔍 Keyword Spotting — exact + semantic matching
  - 📝 Summarization — compressed Arabic summary

---

## 📁 Data Augmentation

Applied to training set only — validation and test kept clean:

| Augmentation | Probability | Simulates |
|---|---|---|
| Gaussian noise | 80% | Microphone background noise |
| Time stretch ±10% | 56% | Fast or slow speakers |
| Pitch shift ±1.5 semitones | 40% | Different voice tones |
| Volume scaling 0.7–1.3x | 80% | Loud or quiet recordings |
| Silence padding up to 0.25s | 24% | Different recording start times |

---

## 👥 Team

| Name |
|---|
|Nour Ezz|

---

## 📄 License

This project is for academic purposes only.

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — base ASR model
- [Google FLEURS](https://huggingface.co/datasets/google/fleurs) — Arabic speech dataset
- [RAVDESS](https://zenodo.org/record/1188976) — emotion speech dataset
- [mT5 XLSum](https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum) — Arabic summarization model
- [HuggingFace Transformers](https://huggingface.co/transformers) — model training framework
- [Gradio](https://gradio.app) — demo interface
