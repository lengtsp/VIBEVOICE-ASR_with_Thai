#!/usr/bin/env python
"""
VibeVoice Enhanced ASR Pipeline
- Noise reduction before inference
- Re-inference for [Speech] segments
- Quality metrics comparison
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import soundfile as sf
import librosa
import torch
import noisereduce as nr

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

# ─── Audio Enhancement ───────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 24000) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32), target_sr


def enhance_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply noise reduction + normalization.
    """
    # 1. Stationary noise reduction (estimate noise from first 0.5s)
    noise_clip_len = min(int(sr * 0.5), len(audio) // 10)
    noise_sample = audio[:noise_clip_len]
    denoised = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_sample,
        stationary=False,
        prop_decrease=0.75,   # 75% noise reduction — preserves speech
        n_fft=2048,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
    )

    # 2. Peak normalization to -1 dBFS
    peak = np.max(np.abs(denoised))
    if peak > 0:
        denoised = denoised * (0.891 / peak)  # -1 dBFS

    return denoised.astype(np.float32)


# ─── Model ───────────────────────────────────────────────────────────────────

def load_model(model_path: str, attn_impl: str = "sdpa"):
    print(f"Loading model from {model_path} ...")
    processor = VibeVoiceASRProcessor.from_pretrained(model_path)
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation=attn_impl,
        trust_remote_code=True,
    ).eval()
    print(f"✅ Model ready on {next(model.parameters()).device}")
    return processor, model


def run_inference(processor, model, audio: np.ndarray, sr: int,
                  context_info: str = "", max_new_tokens: int = 4096) -> dict:
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr, subtype="PCM_16")
    tmp.close()

    inputs = processor(
        audio=tmp.name,
        sampling_rate=sr,
        return_tensors="pt",
        add_generation_prompt=True,
        context_info=context_info,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=processor.pad_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    raw_text = processor.decode(gen_ids, skip_special_tokens=True)
    segments = processor.post_process_transcription(raw_text)

    os.unlink(tmp.name)
    return {"raw_text": raw_text, "segments": segments, "time": elapsed}


# ─── Re-inference for [Speech] segments ──────────────────────────────────────

RETRY_TAGS = {"[Speech]", "[Unintelligible Speech]"}


def enhance_audio_strong(audio: np.ndarray, sr: int) -> np.ndarray:
    """Aggressive noise reduction for hard segments."""
    noise_clip_len = min(int(sr * 0.5), len(audio) // 4)
    noise_sample = audio[:max(noise_clip_len, 1)]
    denoised = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_sample,
        stationary=False,
        prop_decrease=0.90,
        n_fft=2048,
        freq_mask_smooth_hz=300,
        time_mask_smooth_ms=30,
    )
    peak = np.max(np.abs(denoised))
    if peak > 0:
        denoised = denoised * (0.891 / peak)
    return denoised.astype(np.float32)


def retry_speech_segments(processor, model, audio: np.ndarray, sr: int,
                           segments: list, context_info: str = "") -> list:
    """
    For every segment with Content in RETRY_TAGS ('[Speech]' or
    '[Unintelligible Speech]'), slice that audio, apply stronger noise
    reduction, and re-run inference with a tighter context window.
    """
    result = []
    for seg in segments:
        content = seg.get("text", seg.get("Content", ""))
        if content not in RETRY_TAGS:
            result.append(seg)
            continue

        # Extract timestamps
        start = seg.get("start_time", seg.get("Start", 0.0))
        end   = seg.get("end_time",   seg.get("End",   0.0))
        if isinstance(start, str): start = float(start)
        if isinstance(end, str):   end   = float(end)

        # Add ±0.3s padding
        s_idx = max(0, int((start - 0.3) * sr))
        e_idx = min(len(audio), int((end + 0.3) * sr))
        clip  = audio[s_idx:e_idx]
        duration = len(clip) / sr

        if duration < 0.3:
            result.append(seg)
            continue

        tag_label = "Unintelligible" if "[Unintelligible" in content else "Speech"
        print(f"  ↩ Re-inferring [{start:.2f}s – {end:.2f}s] ({duration:.1f}s) [{tag_label}]")
        clip = enhance_audio_strong(clip, sr)
        retry_result = run_inference(
            processor, model, clip, sr,
            context_info=context_info,
            max_new_tokens=512,
        )
        retry_segs = retry_result["segments"]

        if retry_segs:
            # Adjust timestamps back to original timeline
            offset = start - 0.3
            for rs in retry_segs:
                rs_start = rs.get("start_time", rs.get("Start", 0.0))
                rs_end   = rs.get("end_time",   rs.get("End",   0.0))
                if isinstance(rs_start, str): rs_start = float(rs_start)
                if isinstance(rs_end, str):   rs_end   = float(rs_end)
                rs["start_time"] = round(max(start, rs_start + offset), 2)
                rs["end_time"]   = round(min(end,   rs_end   + offset), 2)
            result.extend(retry_segs)
        else:
            result.append(seg)  # keep original if retry also fails

    return result


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(segments: list, total_duration: float) -> dict:
    speech_segs    = [s for s in segments if s.get("text", s.get("Content", "")) not in RETRY_TAGS]
    nonspeech_segs = [s for s in segments if s.get("text", s.get("Content", "")) in RETRY_TAGS]

    def seg_duration(s):
        start = s.get("start_time", s.get("Start", 0.0))
        end   = s.get("end_time",   s.get("End",   0.0))
        if isinstance(start, str): start = float(start)
        if isinstance(end, str):   end   = float(end)
        return max(0.0, end - start)

    transcribed_dur = sum(seg_duration(s) for s in speech_segs)
    missing_dur     = sum(seg_duration(s) for s in nonspeech_segs)

    speakers = set()
    for s in speech_segs:
        sp = s.get("speaker_id", s.get("Speaker"))
        if sp is not None:
            speakers.add(str(sp))

    char_count = sum(len(s.get("text", s.get("Content", ""))) for s in speech_segs)

    return {
        "total_segments":      len(segments),
        "transcribed_segments": len(speech_segs),
        "missing_segments":    len(nonspeech_segs),
        "transcribed_duration_s": round(transcribed_dur, 2),
        "missing_duration_s":  round(missing_dur, 2),
        "coverage_pct":        round(transcribed_dur / max(total_duration, 1) * 100, 1),
        "unique_speakers":     len(speakers),
        "total_chars":         char_count,
    }


def print_metrics(label: str, m: dict):
    print(f"\n{'─'*50}")
    print(f"📊 {label}")
    print(f"{'─'*50}")
    print(f"  Segments total      : {m['total_segments']}")
    print(f"  Transcribed         : {m['transcribed_segments']}")
    print(f"  [Speech] (missing)  : {m['missing_segments']}")
    print(f"  Coverage            : {m['coverage_pct']}%  ({m['transcribed_duration_s']:.1f}s / {m['transcribed_duration_s']+m['missing_duration_s']:.1f}s)")
    print(f"  Unique speakers     : {m['unique_speakers']}")
    print(f"  Total characters    : {m['total_chars']}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    parser = argparse.ArgumentParser(description="VibeVoice Enhanced ASR")
    parser.add_argument("--audio",       required=True, help="Input audio file")
    parser.add_argument("--model_path",  default="/home/indows-11/my_code/model/VibeVoice-ASR")
    parser.add_argument("--attn",        default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--context",     default="", help="Hotwords / context info")
    parser.add_argument("--max_tokens",  type=int, default=4096)
    parser.add_argument("--no_enhance",  action="store_true", help="Skip audio enhancement")
    parser.add_argument("--no_retry",    action="store_true", help="Skip [Speech] retry")
    parser.add_argument("--out",         default=None, help="Save JSON result to file")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  VibeVoice Enhanced ASR Pipeline")
    print(f"{'='*60}")
    print(f"  Audio   : {args.audio}")
    print(f"  Enhance : {not args.no_enhance}")
    print(f"  Retry   : {not args.no_retry}")
    print(f"  Context : {args.context or '(none)'}")
    print(f"{'='*60}\n")

    # Load audio
    print("📂 Loading audio...")
    audio_raw, sr = load_audio(args.audio)
    total_duration = len(audio_raw) / sr
    print(f"  Duration: {total_duration:.1f}s  |  Sample rate: {sr}Hz")

    # ── Baseline run (original audio) ──
    processor, model = load_model(args.model_path, args.attn)

    print("\n[1/3] Running baseline inference (original audio)...")
    baseline = run_inference(processor, model, audio_raw, sr,
                             context_info=args.context,
                             max_new_tokens=args.max_tokens)
    m_baseline = compute_metrics(baseline["segments"], total_duration)
    print_metrics("Baseline (original audio)", m_baseline)
    print(f"  Inference time: {baseline['time']:.1f}s")

    # ── Enhanced run ──
    if not args.no_enhance:
        print("\n[2/3] Applying audio enhancement (noise reduction)...")
        t_enhance = time.time()
        audio_enhanced = enhance_audio(audio_raw, sr)
        print(f"  Enhancement done in {time.time()-t_enhance:.2f}s")

        print("\n       Running inference on enhanced audio...")
        enhanced = run_inference(processor, model, audio_enhanced, sr,
                                 context_info=args.context,
                                 max_new_tokens=args.max_tokens)
        m_enhanced = compute_metrics(enhanced["segments"], total_duration)
        print_metrics("Enhanced (noise-reduced audio)", m_enhanced)
        print(f"  Inference time: {enhanced['time']:.1f}s")

        work_segments = enhanced["segments"]
        work_audio    = audio_enhanced
    else:
        work_segments = baseline["segments"]
        work_audio    = audio_raw

    # ── Retry [Speech] / [Unintelligible Speech] segments ──
    if not args.no_retry:
        retry_count = sum(
            1 for s in work_segments
            if s.get("text", s.get("Content", "")) in RETRY_TAGS
        )
        print(f"\n[3/3] Retrying {retry_count} problematic segment(s)...")
        if retry_count > 0:
            final_segments = retry_speech_segments(
                processor, model, work_audio, sr,
                work_segments, context_info=args.context,
            )
        else:
            final_segments = work_segments
            print("  No problematic segments to retry.")
    else:
        final_segments = work_segments

    m_final = compute_metrics(final_segments, total_duration)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print_metrics("Final result", m_final)

    # Delta vs baseline
    delta_cov = m_final["coverage_pct"] - m_baseline["coverage_pct"]
    delta_missing = m_baseline["missing_segments"] - m_final["missing_segments"]
    print(f"\n  ✨ Coverage improvement : +{delta_cov:.1f}%")
    print(f"  ✨ [Speech] resolved    : {delta_missing} segment(s)")

    # ── Print transcription ──
    print(f"\n{'='*60}")
    print("  TRANSCRIPTION")
    print(f"{'='*60}")
    for seg in final_segments:
        start   = seg.get("start_time", seg.get("Start", "?"))
        end     = seg.get("end_time",   seg.get("End",   "?"))
        speaker = seg.get("speaker_id", seg.get("Speaker", "?"))
        content = seg.get("text",       seg.get("Content", ""))
        print(f"  [{start}s – {end}s] Speaker {speaker}: {content}")

    # ── Save output ──
    output = {
        "audio": args.audio,
        "total_duration_s": total_duration,
        "metrics": {
            "baseline": m_baseline,
            "final":    m_final,
        },
        "segments": final_segments,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Saved to {args.out}")

        total_time = time.time() - t_start
        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

        txt_out = args.out.rsplit(".", 1)[0] + ".txt"
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(f"Audio            : {args.audio}\n")
            f.write(f"Duration         : {total_duration:.1f}s\n")
            f.write(f"Processing time  : {total_time:.1f}s\n")
            f.write(f"GPU memory (peak): {gpu_mem_gb:.2f} GB\n")
            f.write(f"Context          : {args.context or '(none)'}\n")
            f.write("\n")
            for seg in final_segments:
                start   = seg.get("start_time", seg.get("Start", "?"))
                end     = seg.get("end_time",   seg.get("End",   "?"))
                speaker = seg.get("speaker_id", seg.get("Speaker", "?"))
                content = seg.get("text",       seg.get("Content", ""))
                f.write(f"[{start}s – {end}s] Speaker {speaker}: {content}\n")
        print(f"💾 Saved to {txt_out}")

    return output


if __name__ == "__main__":
    main()
