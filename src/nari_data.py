import os
import typing as tp
import numpy as np
import jax
from dataclasses import dataclass
import tensorflow as tf


@dataclass
class DataLoaderConfig:
    """
    Container for dataloader parameters.

    Attributes:
        batch_size: Number of samples per batch.
        text_length: Fixed length for text sequences.
            NOTE: Must be greater than or equal to the maximum text length in the dataset.
        audio_length: Fixed length for audio sequences.
            NOTE: Must be greater than or equal to (maximum audio length in the dataset + max(delay_pattern))
                  to ensure that valid audio data is preserved after applying delays.
        pad_value: Padding value for sequences.
    """

    batch_size: int
    text_length: int
    audio_length: int
    pad_value: int = 0


def create_tf_dataset_from_dir(
    data_dir: str, config: DataLoaderConfig, seed: int = 42
) -> tf.data.Dataset:
    """
    Create a TensorFlow Dataset directly from a directory containing paired .txt and .npy files.
    The dataset is created lazily so that data is read from disk as needed.

    Each sample is processed as follows:
      - The text file is read (as raw bytes) and decoded into a vector of uint8.
      - The corresponding .npy audio file is loaded (via a py_function) into a [None, 9] int16 tensor.
      - Both text and audio are truncated/padded to fixed lengths defined in the config.

    The resulting dataset is shuffled, batched, and prefetched.
    """
    pattern = os.path.join(data_dir, "*.txt")
    ds = tf.data.Dataset.list_files(pattern, shuffle=True, seed=seed)

    def _load_sample(text_path):
        # Read text file and convert to uint8 vector.
        text_content = tf.io.read_file(text_path)
        text = tf.io.decode_raw(text_content, tf.uint8)
        # Compute corresponding .npy audio file path.
        audio_path = tf.strings.regex_replace(text_path, r"\.txt$", ".npy")

        # Use py_function to load .npy file.
        def _load_npy(file_path):
            file_path = file_path.numpy().decode("utf-8")
            audio = np.load(file_path).astype(np.int16)
            return audio

        audio = tf.py_function(func=_load_npy, inp=[audio_path], Tout=tf.int16)
        # Set shape info (audio is [None, 9]).
        audio.set_shape([None, 9])
        return text, audio

    ds = ds.map(_load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    def process_sample(text, audio):
        text = tf.pad(
            text,
            [[0, config.text_length - tf.shape(text)[0]]],
            constant_values=config.pad_value,
        )
        audio = tf.pad(
            audio,
            [[0, config.audio_length - tf.shape(audio)[0]], [0, 0]],
            constant_values=config.pad_value,
        )
        return text, audio

    ds = ds.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


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
    delay_broadcast = delay_arr[None, None, :]  # Shape: (1, 1, C)
    # Compute shifted time indices for each channel.
    new_t = t_idx - delay_broadcast  # Shape: (B, T, C)
    valid = new_t >= 0  # Boolean mask for valid indices.
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
    delay_broadcast = delay_arr[None, None, :]  # Shape: (1, 1, C)
    new_t = t_idx + delay_broadcast  # Shape: (B, T, C)
    valid = new_t < T
    b_idx = np.broadcast_to(np.arange(B)[:, None, None], (B, T, C))
    c_idx = np.broadcast_to(np.arange(C)[None, None, :], (B, T, C))
    result = np.full((B, T, C), pad_value, dtype=delayed_audio.dtype)
    result[valid] = delayed_audio[b_idx[valid], new_t[valid], c_idx[valid]]
    return result
