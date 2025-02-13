import os
import typing as tp
import numpy as np
import jax
from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
    """Container for dataloader parameters.

    Attributes:
        batch_size: Number of samples per batch.
        text_length: Fixed length for text sequences.
            NOTE: Must be greater than or equal to the maximum text length in the dataset.
        audio_length: Fixed length for audio sequences.
            NOTE: Must be greater than or equal to (maximum audio length in the dataset + max(delay_pattern))
                  to ensure that valid audio data is preserved after applying delays.
        g_accum_iters: Gradient accumulation steps.
        pad_value: Padding value for sequences.
    """
    batch_size: int
    text_length: int
    audio_length: int
    g_accum_iters: tp.Optional[int] = None
    pad_value: int = 0


def load_data_pairs(data_dir: str) -> tp.List[tp.Tuple[np.ndarray, np.ndarray]]:
    """Load all text-audio file pairs without length validation."""
    pairs = []
    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    for txt_file in txt_files:
        base = os.path.splitext(txt_file)[0]
        audio_file = os.path.join(data_dir, f"{base}.npy")

        if not os.path.exists(audio_file):
            continue

        # Load text and convert to uint8 tokens.
        with open(os.path.join(data_dir, txt_file), "r", encoding="utf-8") as f:
            text = np.frombuffer(f.read().encode("utf-8"), dtype=np.uint8)

        # Load audio and ensure int16 dtype.
        audio = np.load(audio_file).astype(np.int16)
        pairs.append((text, audio))

    if not pairs:
        raise ValueError(f"No valid text-audio pairs in {data_dir}")
    return pairs


def generate_batch(
    pairs: tp.List[tp.Tuple[np.ndarray, np.ndarray]],
    config: DataLoaderConfig,
    rng_key: jax.Array,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Generate a batch with fixed padding (no truncation).

    This function assumes that the fixed lengths (config.text_length and config.audio_length)
    are chosen to be larger than the maximum lengths in the dataset (for audio, larger than
    max(audio_length) + max(delay_pattern)). Consequently, each sample is padded up to the fixed size.

    Args:
        pairs: All loaded text-audio pairs.
        config: DataLoader configuration.
        rng_key: JAX PRNG key for reproducible sampling.

    Returns:
        Tuple of batches:
          - texts: uint8 array of shape [batch, text_length]
          - audios: int16 array of shape [batch, audio_length, 9]
    """
    bs = config.batch_size * (config.g_accum_iters or 1)
    n_pairs = len(pairs)
    rng_key, choice_key = jax.random.split(rng_key)
    replace = bs > n_pairs
    pair_ix = jax.random.choice(choice_key, n_pairs, shape=(bs,), replace=replace)
    pair_ix_np = np.asarray(pair_ix)

    # Select samples.
    selected_texts = [pairs[i][0] for i in pair_ix_np]
    selected_audios = [pairs[i][1] for i in pair_ix_np]

    # Preallocate full arrays with pad_value.
    text_arr = np.full((bs, config.text_length), config.pad_value, dtype=np.uint8)
    for i, t in enumerate(selected_texts):
        L = t.shape[0]
        text_arr[i, :L] = t

    audio_arr = np.full((bs, config.audio_length, 9), config.pad_value, dtype=np.int16)
    for i, a in enumerate(selected_audios):
        L = a.shape[0]
        audio_arr[i, :L, :] = a

    if config.g_accum_iters:
        text_arr = text_arr.reshape(config.g_accum_iters, config.batch_size, config.text_length)
        audio_arr = audio_arr.reshape(config.g_accum_iters, config.batch_size, config.audio_length, 9)

    return text_arr, audio_arr


def apply_audio_delay(
    audio: np.ndarray,
    pad_value: int,
    delay_pattern: tp.List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8],
) -> np.ndarray:
    """
    Apply a codebook delay pattern to audio tokens without changing the sequence length.

    The input and output audio both have shape [batch, seq_len, 9]. For each codebook channel,
    tokens are shifted right by the specified delay. Positions where valid data is not available
    due to the shift are filled with pad_value. This implementation is fully vectorized.

    Args:
        audio: int16 array of shape [batch, seq_len, 9].
        pad_value: Padding value.
        delay_pattern: List of delay steps for each codebook (must have length 9).

    Returns:
        int16 array of shape [batch, seq_len, 9] with delayed codebooks.
    """
    if len(delay_pattern) != 9:
        raise ValueError("Delay pattern must contain exactly 9 elements")
    B, T, C = audio.shape
    delay_arr = np.array(delay_pattern)  # Shape: (C,)
    # Broadcast time indices to shape (B, T, 1)
    t_idx = np.broadcast_to(np.arange(T)[None, :], (B, T))[:, :, None]  # Shape: (B, T, 1)
    delay_broadcast = delay_arr[None, None, :]       # Shape: (1, 1, C)
    # Compute shifted time indices for each channel.
    new_t = t_idx - delay_broadcast                 # Shape: (B, T, C)
    valid = new_t >= 0                              # Boolean mask for valid indices.
    # Broadcast batch and channel indices to shape (B, T, C)
    b_idx = np.broadcast_to(np.arange(B)[:, None, None], (B, T, C))
    c_idx = np.broadcast_to(np.arange(C)[None, None, :], (B, T, C))
    result = np.full((B, T, C), pad_value, dtype=audio.dtype)
    # Use advanced indexing with the broadcasted indices.
    result[valid] = audio[b_idx[valid], new_t[valid], c_idx[valid]]
    return result


def revert_audio_delay(
    delayed_audio: np.ndarray,
    delay_pattern: tp.List[int],
    pad_value: int,
) -> np.ndarray:
    """
    Reverse the codebook delay pattern to recover original audio tokens without changing the sequence length.

    The input and output audio both have shape [batch, seq_len, 9]. For each codebook channel,
    tokens are shifted left by the specified delay. Positions where valid data is not available
    due to the shift are filled with pad_value. This implementation is fully vectorized.

    Args:
        delayed_audio: int16 array of shape [batch, seq_len, 9].
        delay_pattern: The delay pattern originally applied (must have length 9).
        pad_value: Padding value.

    Returns:
        int16 array of shape [batch, seq_len, 9] with recovered codebooks.
    """
    if len(delay_pattern) != 9:
        raise ValueError("Delay pattern must contain exactly 9 elements")
    B, T, C = delayed_audio.shape
    delay_arr = np.array(delay_pattern)  # Shape: (C,)
    t_idx = np.broadcast_to(np.arange(T)[None, :], (B, T))[:, :, None]  # Shape: (B, T, 1)
    delay_broadcast = delay_arr[None, None, :]       # Shape: (1, 1, C)
    new_t = t_idx + delay_broadcast                 # Shape: (B, T, C)
    valid = new_t < T
    b_idx = np.broadcast_to(np.arange(B)[:, None, None], (B, T, C))
    c_idx = np.broadcast_to(np.arange(C)[None, None, :], (B, T, C))
    result = np.full((B, T, C), pad_value, dtype=delayed_audio.dtype)
    result[valid] = delayed_audio[b_idx[valid], new_t[valid], c_idx[valid]]
    return result
