#!/usr/bin/env python3
"""
Modal GPU function for WhisperX transcription + speaker diarization.

Deploy: modal deploy scripts/modal_transcribe.py
Test:   modal run scripts/modal_transcribe.py
"""

import modal

app = modal.App("cb6-transcribe")

# Build image with all dependencies and pre-downloaded model weights
whisperx_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch",
        "torchaudio",
        "whisperx @ git+https://github.com/m-bain/whisperX.git",
        "pyannote.audio>=3.1",
        "transformers",
        "accelerate",
    )
    # Pre-download model weights at image build time (cached across runs)
    .run_commands(
        "python -c \"import whisperx; whisperx.load_model('distil-large-v3', 'cpu', compute_type='float32')\"",
    )
)


@app.function(
    image=whisperx_image,
    gpu="T4",
    timeout=1800,  # 30 min max per transcription
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def transcribe(audio_bytes: bytes, num_speakers: int = None) -> dict:
    """Transcribe audio bytes using WhisperX with speaker diarization.

    Args:
        audio_bytes: Raw audio file bytes (WAV or any ffmpeg-supported format)
        num_speakers: Optional hint for number of speakers (None = auto-detect)

    Returns:
        dict with 'segments' (list of segments with speaker labels and word timestamps)
    """
    import os
    import tempfile

    import whisperx

    device = "cuda"
    compute_type = "float16"

    # Write audio bytes to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    try:
        # 1. Transcribe with distil-large-v3
        model = whisperx.load_model(
            "distil-large-v3", device, compute_type=compute_type
        )
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)

        # Free GPU memory
        del model
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

        # 2. Align whisper output for word-level timestamps
        model_a, metadata = whisperx.load_align_model(
            language_code="en", device=device
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device
        )

        del model_a
        gc.collect()
        torch.cuda.empty_cache()

        # 3. Speaker diarization (using pyannote/speaker-diarization-3.1)
        from whisperx.diarize import DiarizationPipeline

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            diarize_model = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                token=hf_token,
                device=device,
            )
            diarize_kwargs = {}
            if num_speakers is not None:
                diarize_kwargs["num_speakers"] = num_speakers
            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        # 4. Clean up and return
        segments = []
        for seg in result["segments"]:
            segment = {
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg["text"].strip(),
                "speaker": seg.get("speaker", "UNKNOWN"),
            }
            if "words" in seg:
                segment["words"] = [
                    {
                        "word": w.get("word", ""),
                        "start": round(w.get("start", 0), 3),
                        "end": round(w.get("end", 0), 3),
                        "score": round(w.get("score", 0), 3),
                    }
                    for w in seg["words"]
                    if "word" in w
                ]
            segments.append(segment)

        return {"segments": segments}

    finally:
        os.unlink(audio_path)


# For testing: modal run scripts/modal_transcribe.py
@app.local_entrypoint()
def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: modal run scripts/modal_transcribe.py -- <audio_file>")
        print("Example: modal run scripts/modal_transcribe.py -- audio/test.wav")
        return

    audio_path = sys.argv[1]
    print(f"Reading audio from {audio_path}...")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    print(f"Sending {len(audio_bytes) / 1024 / 1024:.1f} MB to Modal...")
    result = transcribe.remote(audio_bytes)

    print(f"Got {len(result['segments'])} segments")
    for seg in result["segments"][:5]:
        speaker = seg.get("speaker", "?")
        print(f"  [{speaker}] {seg['start']:.1f}s: {seg['text'][:80]}")

    import json

    with open("test_transcription.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull result saved to test_transcription.json")
