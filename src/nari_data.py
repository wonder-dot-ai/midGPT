import os
import typing as tp
import numpy as np
import jax
from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
    """Container for dataloader parameters

    Attributes:
        batch_size: Number of samples per batch
        g_accum_iters: Gradient accumulation steps
        pad_value: Padding value for sequences
    """

    batch_size: int
    g_accum_iters: tp.Optional[int] = None
    pad_value: int = 0


def load_data_pairs(data_dir: str) -> tp.List[tp.Tuple[np.ndarray, np.ndarray]]:
    """Load all text-audio file pairs without length validation"""
    pairs = []
    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    for txt_file in txt_files:
        base = os.path.splitext(txt_file)[0]
        audio_file = os.path.join(data_dir, f"{base}.npy")

        if not os.path.exists(audio_file):
            continue

        # Load text
        with open(os.path.join(data_dir, txt_file), "r", encoding="utf-8") as f:
            text = np.frombuffer(f.read().encode("utf-8"), dtype=np.uint8)

        # Load audio
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
    """Generate batch with dynamic padding to longest in batch

    Args:
        pairs: All loaded text-audio pairs
        config: DataLoader configuration
        rng_key: JAX PRNG key for reproducible sampling

    Returns:
        Tuple of padded batches where:
        - texts: uint8 array [batch, max_text_len]
        - audios: int16 array [batch, max_audio_len, 9]
    """
    bs = config.batch_size * (config.g_accum_iters or 1)
    n_pairs = len(pairs)

    # Split RNG key for sampling operations
    rng_key, choice_key = jax.random.split(rng_key)

    # Smart selection with replacement only when necessary
    replace = bs > n_pairs  # Only replace if we need more samples than available
    pair_ix = jax.random.choice(choice_key, n_pairs, shape=(bs,), replace=replace)

    # Convert to numpy for indexing
    pair_ix_np = np.asarray(pair_ix)

    # Collect selected samples
    selected_texts = [pairs[i][0] for i in pair_ix_np]
    selected_audios = [pairs[i][1] for i in pair_ix_np]

    # Find maximum lengths in this batch
    max_text_len = max(len(t) for t in selected_texts)
    max_audio_len = max(len(a) for a in selected_audios)

    # Pad texts
    text_batch = [
        np.pad(t, (0, max_text_len - len(t)), constant_values=config.pad_value)
        for t in selected_texts
    ]

    # Pad audios (preserve codebook dimension)
    audio_batch = [
        np.pad(
            a, [(0, max_audio_len - len(a)), (0, 0)], constant_values=config.pad_value
        )
        for a in selected_audios
    ]

    # Convert to arrays
    text_arr = np.array(text_batch, dtype=np.uint8)
    audio_arr = np.array(audio_batch, dtype=np.int16)

    # Reshape for gradient accumulation
    if config.g_accum_iters:
        text_arr = text_arr.reshape(config.g_accum_iters, config.batch_size, -1)
        audio_arr = audio_arr.reshape(config.g_accum_iters, config.batch_size, -1, 9)

    return text_arr, audio_arr


def apply_audio_delay(
    audio: np.ndarray,
    pad_value: int,
    delay_pattern: tp.List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8],
) -> np.ndarray:
    """
    Apply codebook delay pattern to audio tokens with padding

    Args:
        audio: int16 array of shape [batch, seq_len, 9]
        pad_value: Value used for padding delayed sequences
        delay_pattern: List of delay steps for each codebook (length must match 9)

    Returns:
        int16 array of shape [batch, seq_len + max(delay), 9] with delayed codebooks
    """
    if len(delay_pattern) != 9:
        raise ValueError("Delay pattern must contain exactly 9 elements")

    batch_size, seq_len, _ = audio.shape
    max_delay = max(delay_pattern)
    new_seq_len = seq_len + max_delay

    # Initialize output array with padding
    delayed = np.full((batch_size, new_seq_len, 9), pad_value, dtype=audio.dtype)

    for cb_idx, delay in enumerate(delay_pattern):
        # Calculate padding for this codebook
        right_pad = max_delay - delay

        # Pad and shift the codebook tokens
        padded = np.pad(
            audio[..., cb_idx],
            [(0, 0), (delay, right_pad)],  # (left, right) padding on time axis
            mode="constant",
            constant_values=pad_value,
        )

        # Insert into output array
        delayed[..., cb_idx] = padded

    return delayed


def revert_audio_delay(
    delayed_audio: np.ndarray,
    delay_pattern: tp.List[int],
    pad_value: int,
) -> np.ndarray:
    """
    Reverse codebook delay pattern to recover original audio tokens

    Args:
        delayed_audio: int16 array of shape [batch, delayed_seq_len, 9]
        delay_pattern: Original delay pattern used for shifting
        pad_value: Padding value used in delayed audio

    Returns:
        int16 array of shape [batch, original_seq_len, 9] with aligned codebooks
    """
    if len(delay_pattern) != 9:
        raise ValueError("Delay pattern must contain exactly 9 elements")

    batch_size, delayed_seq_len, _ = delayed_audio.shape
    max_delay = max(delay_pattern)
    original_seq_len = delayed_seq_len - max_delay

    if original_seq_len <= 0:
        raise ValueError("Invalid delayed audio sequence length")

    # Initialize output array
    reverted = np.full(
        (batch_size, original_seq_len, 9), pad_value, dtype=delayed_audio.dtype
    )

    for cb_idx, delay in enumerate(delay_pattern):
        # Calculate valid slice positions
        start = delay
        end = start + original_seq_len

        # Extract original codebook sequence
        codebook_slice = delayed_audio[:, start:end, cb_idx]

        # Handle any remaining padding
        valid_mask = codebook_slice != pad_value
        reverted[..., cb_idx] = np.where(valid_mask, codebook_slice, pad_value)

    return reverted
