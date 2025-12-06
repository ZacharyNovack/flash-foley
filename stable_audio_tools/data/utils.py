import math
import random
import torch

from torch import nn
from typing import Tuple

from torchaudio import transforms as T
import librosa
import warnings
import torchaudio
from pesto import load_model as pesto_load_model
import scipy
MIN_DB = -100.
REF_DB = 20.
CREPE_SR = 16000
CREPE_WIN_SIZE = 1024

def get_perceptual_weights_cpu():
    """Compute A-weighting curve once using librosa (on CPU)"""
    freqs = librosa.fft_frequencies(sr=CREPE_SR, n_fft=CREPE_WIN_SIZE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        a_weight = librosa.A_weighting(freqs) - REF_DB
    return torch.tensor(a_weight, dtype=torch.float32)  # shape (F,)

def perceptual_weights():
    """A-weighted frequency-dependent perceptual loudness weights"""
    frequencies = librosa.fft_frequencies(sr=CREPE_SR,
                                          n_fft=CREPE_WIN_SIZE)

    # A warning is raised for nearly inaudible frequencies, but it ends up
    # defaulting to -100 db. That default is fine for our purposes.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return librosa.A_weighting(frequencies)[:, None] - REF_DB


# Cache CPU-computed weights (computed once)
_A_WEIGHT_CPU = get_perceptual_weights_cpu()[:, None]  # shape (F, 1)
hann_window = torch.tensor(scipy.signal.get_window('hann', CREPE_WIN_SIZE, fftbins=True))


def a_weighted_gpu(audio: torch.Tensor, sample_rate: int, hop_length=None, pad=True) -> torch.Tensor:
    """
    GPU-based A-weighted loudness estimation using torchaudio and precomputed perceptual weights.
    
    Args:
        audio: Tensor of shape (B, C, T), on GPU
        sample_rate: Sampling rate of audio
        hop_length: Hop size (defaults to 10ms)
        pad: Whether to center the STFT
    
    Returns:
        Tensor of shape (1, n_frames) representing loudness in dB
    """

    if not hasattr(a_weighted_gpu, "weights"):
        a_weighted_gpu.weights = _A_WEIGHT_CPU.to(audio.device)
        a_weighted_gpu.window = hann_window.to(audio.device)
    

    if audio.ndim == 3:
        audio = audio.mean(dim=1)
    audio = audio.to(torch.float32)  
    device = audio.device
    hop_length = sample_rate // 100 if hop_length is None else hop_length

    # Resample if needed
    if sample_rate != CREPE_SR:
        audio = torchaudio.functional.resample(
            waveform=audio,
            orig_freq=sample_rate,
            new_freq=CREPE_SR
        )
        hop_length = int(hop_length * CREPE_SR / sample_rate)
        sample_rate = CREPE_SR

    # Compute STFT
    stft = torch.stft(audio,
                      n_fft=CREPE_WIN_SIZE,
                      hop_length=hop_length,
                      win_length=CREPE_WIN_SIZE,
                      window=a_weighted_gpu.window,
                      center=pad,
                      normalized=False,
                      pad_mode='constant',
                      return_complex=True)  # (B, F, T)
    mag = stft.abs().pow(2)

    # Convert to dB
    db = 10.0 * torch.log10(torch.maximum(torch.tensor(1e-10), mag))
    db = torch.maximum(db, db.max() - 80)

    # Apply precomputed perceptual weights
    weights = a_weighted_gpu.weights
    weighted_db = db + weights

    # Threshold
    weighted_db = torch.clamp(weighted_db, min=MIN_DB)

    # Mean across frequency bins
    loudness = weighted_db.mean(dim=1, keepdim=True)  # (B, 1, T)
    return loudness, stft

def pesto_pitch_256(audio, sr=44100, hop_length=3000):
    # pesto expects (B, T) in float32, mono
    if audio.ndim == 3 and audio.shape[1] > 1:
        audio = audio.mean(dim=1)  # (B, T)
    elif audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    # if getattr(pesto_pitch, "model", None) is None:
    with torch.no_grad():
        if not hasattr(pesto_pitch_256, "model"):
            print("Loading pesto pitch extraction model")
            # Load the pesto model
            sr = int(sr)
            pesto_pitch_256.model = pesto_load_model("mir-1k_g7", step_size=20, sampling_rate=sr).to(audio.device)
        model = pesto_pitch_256.model
        # pesto expects shape (B, T)
        _, _2, _3, activations = model(audio, sr=sr)
        # downsample to 256 Hz
        # activations = torch.nn.functional.interpolate(activations, size=(256,), mode='linear', align_corners=False)
        return activations  # shape: (B, frames, num_classes)

def spectral_centroid(audio, sr=44100, n_fft=2048, hop_length=512):
    # Compute STFT frequencies
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    B, C, T = audio.shape

    # Mixdown to mono
    audio_mono = audio.mean(dim=1)  # (B, T)

    # Compute STFT
    window = torch.hann_window(n_fft, device=audio.device)
    spectrogram = torch.stft(audio_mono, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)  # (B, F, frames)
    spectrogram = spectrogram.abs()  # (B, F, frames)

    frequencies = torch.abs(torch.fft.fftfreq(
        2 * (spectrogram.shape[0] - 1),
        1 / sr,
        device=spectrogram.device
    )[:spectrogram.shape[0]])

    # Compute centroid
    centroid = (
        frequencies * spectrogram.T
    ).sum(dim=1) / spectrogram.sum(dim=0).squeeze()
    centroid_midi = 69 + 12 * torch.log2(centroid / 440.0 + 1e-10)
    centroid_midi = torch.clamp(centroid_midi, min=0, max=127)
    # Scale to (0, 1) by dividing by 127 (G9)
    centroid_scaled = centroid_midi / 127.0
    return centroid_scaled  # (frames,)

class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

class PadCrop_Normalized_T(nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, source: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int]:
        
        n_channels, n_samples = source.shape
        
        # If the audio is shorter than the desired length, pad it
        upper_bound = max(0, n_samples - self.n_samples)
        
        # If randomize is False, always start at the beginning of the audio
        offset = 0
        if(self.randomize and n_samples > self.n_samples):
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)

        # Create the chunk
        chunk = source.new_zeros([n_channels, self.n_samples])

        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = source[:, offset:offset + self.n_samples]
        
        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        
        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )

class PhaseFlipper(nn.Module):
    "Randomly invert the phase of a signal"
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def __call__(self, signal):
        return -signal if (random.random() < self.p) else signal
        
class Mono(nn.Module):
  def __call__(self, signal):
    return torch.mean(signal, dim=0, keepdims=True) if len(signal.shape) > 1 else signal

class Stereo(nn.Module):
  def __call__(self, signal):
    signal_shape = signal.shape
    # Check if it's mono
    if len(signal_shape) == 1: # s -> 2, s
        signal = signal.unsqueeze(0).repeat(2, 1)
    elif len(signal_shape) == 2:
        if signal_shape[0] == 1: #1, s -> 2, s
            signal = signal.repeat(2, 1)
        elif signal_shape[0] > 2: #?, s -> 2,s
            signal = signal[:2, :]    

    return signal

class VolumeNorm(nn.Module):
    "Volume normalization and augmentation of a signal [LUFS standard]"
    def __init__(self, params=[-16, 2], sample_rate=16000, energy_threshold=1e-6):
        super().__init__()
        self.loudness = T.Loudness(sample_rate)
        self.value = params[0]
        self.gain_range = [-params[1], params[1]]
        self.energy_threshold = energy_threshold

    def __call__(self, signal):
        """
        signal: torch.Tensor [channels, time]
        """
        # avoid do normalisation for silence
        energy = torch.mean(signal**2)
        if energy < self.energy_threshold:
            return signal
        
        input_loudness = self.loudness(signal)
        # Generate a random target loudness within the specified range
        target_loudness = self.value + (torch.rand(1).item() * (self.gain_range[1] - self.gain_range[0]) + self.gain_range[0])
        delta_loudness = target_loudness - input_loudness
        gain = torch.pow(10.0, delta_loudness / 20.0)
        output = gain * signal

        # Check for potentially clipped samples
        if torch.max(torch.abs(output)) >= 1.0:
            output = self.declip(output)

        return output

    def declip(self, signal):
        """
        Declip the signal by scaling down if any samples are clipped
        """
        max_val = torch.max(torch.abs(signal))
        if max_val > 1.0:
            signal = signal / max_val
            signal *= 0.95
        return signal

        
        

