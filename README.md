# VibeVoice ASR

Thai/multilingual speech-to-text with speaker diarization and noise reduction.

> **Source:** [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) on Hugging Face

---

## Requirements

- Python 3.11
- CUDA-capable GPU (CUDA 12.x or 13.x)
- Conda

---

## Installation

**1. Create environment**

```bash
conda create -n vibevoice_asr python=3.11
conda activate vibevoice_asr
```

**2. Install PyTorch** — match your CUDA version

```bash
# CUDA 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**3. Install dependencies**

```bash
pip install -r requirements_vibevoice_asr.txt
```

**4. Install VibeVoice library**

```bash
cd /path/to/VibeVoice
pip install -e .
```

---

## Model Weights

Download or copy the model weights (`VibeVoice-ASR/`) to your local path.  
Required files: `config.json` + `model-*.safetensors` (~14 GB total)

---

## Run

```bash
cd /path/to/VibeVoice
conda activate vibevoice_asr

python demo/vibevoice_enhanced_asr.py \
  --audio "path/to/audio.mp3" \
  --model_path /path/to/VibeVoice-ASR \
  --attn sdpa \
  --context "keyword1, keyword2, proper-noun" \
  --out result.json
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--audio` | required | Path to input audio file |
| `--model_path` | required | Path to VibeVoice-ASR weights |
| `--attn` | `sdpa` | Attention: `sdpa` or `flash_attention_2` |
| `--context` | — | Hotwords (names, org names, technical terms) |
| `--out` | `result.json` | Output JSON path |
| `--no_enhance` | — | Skip noise reduction |
| `--no_retry` | — | Skip retry on `[Speech]` segments |
| `--max_tokens` | `4096` | Max generation tokens |

### Context Tips

Use `--context` for **proper nouns and technical terms only** — not common words.

```bash
# Good
--context "John Smith, Acme Corp, cybercrime, mule account"

# Bad — causes hallucination
--context "said, showed, arrested"
```

---

## Output

Results are saved as `result.json` and `result.txt`.

```json
{
  "audio": "path/to/audio.mp3",
  "total_duration_s": 157.6,
  "processing_time_s": 40.1,
  "gpu_memory_peak_gb": 19.69,
  "n_speakers": 3,
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 12.69,
      "speaker_id": 0,
      "text": "transcribed content..."
    }
  ]
}
```

---

## Other Scripts

| Script | Description |
|---|---|
| `vibevoice_enhanced_asr.py` | **Best text quality** — noise reduction + retry |
| `vibevoice_native_diarize.py` | Fast native speaker diarization |
| `vibevoice_diarize.py` | Accurate speaker diarization via pyannote |
| `vibevoice_asr_gradio_demo.py` | Web UI (Gradio) |
