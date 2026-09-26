"""
Audio processing service for HealHub Voice Emotion Demo.
Converts raw audio bytes into 16kHz single-channel (mono) float32 numpy array.
"""

import io
import wave
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def prepare_audio_array(audio_bytes: bytes) -> np.ndarray:
    """
    Decodes audio bytes (WAV, OGG, FLAC, MP3, etc.) into mono float32 numpy array at 16kHz.
    
    Time Complexity: O(N) where N is number of audio samples.
    Space Complexity: O(N) memory allocation for target audio array.
    """
    if not audio_bytes:
        raise ValueError("Audio bytes are empty")

    # 1. Primary decoder: SoundFile (handles standard WAV, OGG, FLAC, AIFF)
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != TARGET_SR:
            gcd = math.gcd(sr, TARGET_SR)
            data = resample_poly(data, TARGET_SR // gcd, sr // gcd).astype(np.float32)
        return data
    except Exception as sf_err:
        pass

    # 2. Secondary decoder: Standard Library Python wave module (handles 8, 16, 32-bit PCM WAV)
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
            
            if sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 4:
                dtype = np.int32
            else:
                dtype = np.uint8
                
            data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            if dtype == np.int16:
                data /= 32768.0
            elif dtype == np.int32:
                data /= 2147483648.0
            elif dtype == np.uint8:
                data = (data - 128.0) / 128.0

            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)

            if sr != TARGET_SR:
                gcd = math.gcd(sr, TARGET_SR)
                data = resample_poly(data, TARGET_SR // gcd, sr // gcd).astype(np.float32)

            return data
    except Exception as wave_err:
        pass

    # 3. Tertiary fallback: Torchaudio if available
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
        raise RuntimeError(f"Could not decode audio file format. Please ensure audio is WAV, OGG, or MP3.")
