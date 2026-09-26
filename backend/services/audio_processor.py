"""
Audio processing service for HealHub Voice Emotion Demo.
Converts raw audio bytes into 16kHz single-channel (mono) float32 numpy array.
"""

import io
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import math

TARGET_SR = 16000


def prepare_audio_array(audio_bytes: bytes) -> np.ndarray:
    """
    Decodes audio bytes (WAV, OGG, FLAC, MP3, etc.) into mono float32 numpy array at 16kHz.
    
    Time Complexity: O(N) where N is number of audio samples.
    Space Complexity: O(N) memory allocation for target audio array.
    """
    if not audio_bytes:
        raise ValueError("Audio bytes are empty")

    try:
        # soundfile handles WAV, OGG, FLAC, etc.
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        
        # Convert multi-channel to mono if necessary
        if data.ndim > 1:
            data = np.mean(data, axis=1)
            
        # Resample to 16kHz if necessary
        if sr != TARGET_SR:
            gcd = math.gcd(sr, TARGET_SR)
            up = TARGET_SR // gcd
            down = sr // gcd
            data = resample_poly(data, up, down).astype(np.float32)
            
        return data
    except Exception as sf_err:
        # Fallback to torchaudio or raw array if soundfile fails on non-standard containers
        try:
            import torchaudio
            import torch
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                waveform = resampler(waveform)
            return waveform.squeeze(0).numpy().astype(np.float32)
        except Exception as torch_err:
            raise RuntimeError(f"Could not decode audio file format. Soundfile error: {sf_err}; Torchaudio error: {torch_err}")
