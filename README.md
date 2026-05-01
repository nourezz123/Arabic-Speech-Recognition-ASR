# 🎙️ Arabic Automatic Speech Recognition (ASR)
## Deep Learning Based Arabic Audio Understanding and Retrieval System

> An end-to-end Arabic Speech Recognition pipeline using OpenAI Whisper Medium, evaluated on two Arabic speech datasets with full EDA, WER/CER evaluation, and an interactive Gradio demo.

---

## 📊 Results

| Metric | Arabic Speech Corpus | Common Voice Arabic |
|--------|---------------------|---------------------|
| **Samples** | 100 | 100 |
| **WER** | N/A (phonetic refs) | 52.17% |
| **CER** | N/A (phonetic refs) | 21.46% |
| **Word Accuracy** | Visually accurate ✅ | 47.83% |
| **Avg Duration** | 10.30s | 4.52s |

---

## 📌 Project Overview

This project implements a complete Arabic ASR system using **OpenAI Whisper Medium** (307M parameters), a transformer-based model trained on 680,000 hours of multilingual speech. The pipeline covers exploratory data analysis, audio preprocessing, transcription, WER/CER evaluation, and an interactive demo interface.

**Two notebooks:**
- `eda.ipynb` — Exploratory Data Analysis (waveforms, spectrograms, statistics)
- `Main pipeline.ipynb` — Full ASR pipeline, evaluation, and Gradio demo

---

## 🏗️ System Architecture

```
Audio Input (.wav / .mp3)
        │
        ▼
Preprocessing
(Resampling → 16kHz, Normalization)
        │
        ▼
Feature Extraction
(Log-Mel Spectrogram — 80 bins, 25ms window)
        │
        ▼
Whisper Medium Model
(CNN Stem → Transformer Encoder → Transformer Decoder)
        │
        ▼
Arabic Text Output (Unicode)
        │
        ▼
Evaluation (WER / CER)
```

### Model Components

| Component | Role |
|-----------|------|
| **CNN Stem** | Extracts local acoustic features from the spectrogram |
| **Transformer Encoder** | Captures long-range audio dependencies |
| **Transformer Decoder** | Auto-regressively generates Arabic text tokens |

---

## 📁 Datasets

### Dataset 1 — Arabic Speech Corpus

| Property | Value |
|----------|-------|
| **Source** | [Arabic Speech Corpus](https://en.arabicspeechcorpus.com/) |
| **Download** | https://en.arabicspeechcorpus.com/ |
| **Speaker** | Single native MSA speaker |
| **Quality** | Studio-recorded (high SNR) |
| **Format** | WAV, 16kHz |
| **Test Samples** | 100 sentences |
| **References** | Buckwalter phonetic transliteration |
| **License** | Free for research use |

### Dataset 2 — Mozilla Common Voice Arabic (v25.0)

| Property | Value |
|----------|-------|
| **Source** | [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets) |
| **Download** | https://commonvoice.mozilla.org/en/datasets → Select Arabic (ar) |
| **Speakers** | Multiple crowd-sourced speakers |
| **Quality** | Variable (consumer microphones) |
| **Format** | MP3, 48kHz |
| **Test Samples** | 10,508 available (100 used) |
| **References** | Arabic script ✅ (real WER calculation) |
| **License** | CC-0 (Public Domain) |

> **Note:** After downloading Common Voice, extract and place at:
> `data/common_voice/<extracted_folder>/cv-corpus-25.0-2026-03-09/ar/`

---

## 🗂️ Project Structure

```
Arabic-Speech-Recognition-ASR/
│
├── eda.ipynb                  ← Exploratory Data Analysis
├── Main pipeline.ipynb        ← Full ASR pipeline + Demo
│
├── asc_transcriptions.csv     ← Arabic Speech Corpus results
├── cv_transcriptions.csv      ← Common Voice results + WER
├── evaluation_summary.csv     ← Final metrics summary
│
├── eda_waveform_asc.png       ← Waveform & spectrogram plots
├── eda_comparison.png         ← Dataset audio comparison
├── eda_statistics.png         ← Duration & word count stats
├── eda_word_freq.png          ← Word frequency analysis
├── eda_quality.png            ← Signal quality analysis
├── evaluation_results.png     ← WER/CER charts
├── error_analysis.png         ← Error breakdown
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- FFmpeg

### Install FFmpeg (Windows)
1. Download: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
2. Extract → move to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to system PATH
4. Verify: `ffmpeg -version`

### Install Python Dependencies

```bash
pip install openai-whisper torch torchaudio datasets transformers jiwer gradio soundfile matplotlib seaborn librosa arabic-reshaper python-bidi pandas numpy scipy
```

---

## 📥 Dataset Setup

### Dataset 1 — Arabic Speech Corpus
1. Go to https://en.arabicspeechcorpus.com/
2. Download the corpus
3. Extract to: `data/arabic_speech/arabic-speech-corpus/`

### Dataset 2 — Mozilla Common Voice Arabic
1. Go to https://commonvoice.mozilla.org/en/datasets
2. Find **Arabic (ar)** → enter email → Download
3. Extract to: `data/common_voice/`

---

## 🚀 Usage

### 1. Run EDA
```bash
jupyter notebook eda.ipynb
```

### 2. Run Main Pipeline
```bash
jupyter notebook "Main pipeline.ipynb"
```

### 3. Interactive Demo
The last cell of `Main pipeline.ipynb` launches a Gradio interface:
- Upload any Arabic `.wav` or `.mp3` file
- Or record from your microphone
- Get instant Arabic transcription

---

## 📈 Visualizations

| Plot | Description |
|------|-------------|
| `eda_waveform_asc.png` | Waveform, Mel spectrogram, MFCC |
| `eda_comparison.png` | ASC vs Common Voice audio quality |
| `eda_statistics.png` | Duration & word count distributions |
| `eda_word_freq.png` | Top Arabic words frequency |
| `eda_quality.png` | SNR & signal energy analysis |
| `evaluation_results.png` | WER/CER charts & model comparison |
| `error_analysis.png` | Prediction quality breakdown |

---

## 📦 Requirements

```
openai-whisper
torch
torchaudio
datasets
jiwer
gradio
soundfile
matplotlib
seaborn
librosa
arabic-reshaper
python-bidi
pandas
numpy
scipy
```

---

## ⚠️ Notes

- Running on **CPU** takes ~15-25 min per 100 files. GPU reduces this to ~2-3 min.
- The Arabic Speech Corpus references are in **Buckwalter phonetic format** — WER is calculated on Common Voice Arabic only (Arabic script references).
- WER of 52.17% on crowd-sourced dialectal Arabic is expected without fine-tuning — CER of 21.46% shows character-level accuracy is much better.

---

## 📚 References

- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Radford et al., OpenAI (2022)
- [Arabic Speech Corpus](https://en.arabicspeechcorpus.com) — Halabi (2016)
- [Mozilla Common Voice](https://commonvoice.mozilla.org) — Mozilla Foundation
- [jiwer — WER/CER computation](https://github.com/jitsi/jiwer)

---

## 👤 Author

**Nour Ezz**
