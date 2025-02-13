import pathlib
import typing as tp
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class DataLoaderConfig:
    """
    DataLoader configuration.

    Attributes:
      batch_size (int): Number of samples per batch.
      text_length (int): Fixed text length (each text tensor becomes [text_length]).
      audio_length (int): Fixed audio length (each audio tensor becomes [audio_length, 9]).
      pad_value (int): Padding value.
      delay_pattern (List[int]): Delay steps for 9 codebook channels.
    """
    batch_size: int
    text_length: int
    audio_length: int
    pad_value: int = 0
    delay_pattern: tp.List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8])

def create_dataset(data_dir: pathlib.Path, config: DataLoaderConfig, seed: int = 42) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from paired .txt and .npy files.

    Each sample:
      - Reads a text file, decodes it to a uint8 vector (shape [L_text]),
      - Loads a .npy audio file as an int16 tensor (shape [T, 9]),
      - Pads text to [text_length] and audio to [audio_length, 9],
      - Applies an audio delay using config.delay_pattern.

    Args:
      data_dir (pathlib.Path): Directory with .txt and .npy files.
      config (DataLoaderConfig): Data loader configuration.
      seed (int): Shuffle seed.

    Returns:
      tf.data.Dataset: Yields tuples (txt_L, delayed_audio_L9) where:
        txt_L: tf.Tensor, shape [batch_size, text_length], dtype tf.uint8.
        delayed_audio_L9: tf.Tensor, shape [batch_size, audio_length, 9], dtype tf.int16.
    """
    pattern = str(data_dir / "*.txt")
    ds = tf.data.Dataset.list_files(pattern, shuffle=True, seed=seed)

    def _load_sample(text_path: tf.Tensor) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        txt_1D = tf.io.decode_raw(tf.io.read_file(text_path), tf.uint8)
        audio_path = tf.strings.regex_replace(text_path, r"\.txt$", ".npy")
        
        def _load_npy(fp: tf.Tensor) -> np.ndarray:
            fp_str = fp.numpy().decode("utf-8")
            return np.load(fp_str).astype(np.int16)
        audio_T9 = tf.py_function(_load_npy, inp=[audio_path], Tout=tf.int16)
        audio_T9.set_shape([None, 9])
        return txt_1D, audio_T9

    ds = ds.map(_load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    def process_sample(txt_1D: tf.Tensor, audio_T9: tf.Tensor) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        txt_L = tf.pad(txt_1D, [[0, config.text_length - tf.shape(txt_1D)[0]]], constant_values=config.pad_value)
        audio_L9 = tf.pad(audio_T9, [[0, config.audio_length - tf.shape(audio_T9)[0]], [0, 0]], constant_values=config.pad_value)
        return txt_L, audio_L9

    ds = ds.map(process_sample, num_parallel_calls=tf.data.AUTOTUNE).cache()

    def apply_delay(txt_L: tf.Tensor, audio_L9: tf.Tensor) -> tp.Tuple[tf.Tensor, tf.Tensor]:
        delayed_audio_L9 = tf.py_function(
            func=lambda aud: apply_audio_delay(aud, config.pad_value, config.delay_pattern),
            inp=[audio_L9],
            Tout=tf.int16)
        delayed_audio_L9.set_shape(audio_L9.shape)
        return txt_L, delayed_audio_L9

    ds = ds.map(apply_delay, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def apply_audio_delay(
    audio_BTC: np.ndarray,
    pad_value: int,
    delay_pattern: tp.List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
) -> np.ndarray:
    """
    Applies a delay pattern to audio tokens.

    Args:
      audio_BTC (np.ndarray): int16 array with shape [B, T, 9] 
         (B=batch size, T=audio frames, 9=channels).
      pad_value (int): Padding value.
      delay_pattern (List[int]): Delay steps for each channel (length 9).

    Returns:
      np.ndarray: Delayed audio with shape [B, T, 9], dtype int16.
    """
    if len(delay_pattern) != 9:
        raise ValueError("Delay pattern must have 9 elements")
    B, T, C = audio_BTC.shape
    delay_arr_C = np.array(delay_pattern)
    t_idx_BTx1 = np.broadcast_to(np.arange(T)[None, :], (B, T))[:, :, None]
    new_t_BTC = t_idx_BTx1 - delay_arr_C[None, None, :]
    valid_BTC = new_t_BTC >= 0
    b_idx_BTC = np.broadcast_to(np.arange(B)[:, None, None], (B, T, C))
    c_idx_BTC = np.broadcast_to(np.arange(C)[None, None, :], (B, T, C))
    result_BTC = np.full((B, T, C), pad_value, dtype=audio_BTC.dtype)
    result_BTC[valid_BTC] = audio_BTC[b_idx_BTC[valid_BTC], new_t_BTC[valid_BTC], c_idx_BTC[valid_BTC]]
    return result_BTC

def revert_audio_delay(
    delayed_audio_BTC: np.ndarray,
    delay_pattern: tp.List[int],
    pad_value: int
) -> np.ndarray:
    """
    Reverts a delay pattern from audio tokens.

    Args:
      delayed_audio_BTC (np.ndarray): int16 array with shape [B, T, 9].
      delay_pattern (List[int]): Delay pattern (length 9).
      pad_value (int): Padding value.
    
    Returns:
      np.ndarray: Recovered audio with shape [B, T, 9], dtype int16.
    """
    if len(delay_pattern) != 9:
        raise ValueError("Delay pattern must have 9 elements")
    B, T, C = delayed_audio_BTC.shape
    delay_arr_C = np.array(delay_pattern)
    t_idx_BTx1 = np.broadcast_to(np.arange(T)[None, :], (B, T))[:, :, None]
    new_t_BTC = t_idx_BTx1 + delay_arr_C[None, None, :]
    valid_BTC = new_t_BTC < T
    b_idx_BTC = np.broadcast_to(np.arange(B)[:, None, None], (B, T, C))
    c_idx_BTC = np.broadcast_to(np.arange(C)[None, None, :], (B, T, C))
    result_BTC = np.full((B, T, C), pad_value, dtype=delayed_audio_BTC.dtype)
    result_BTC[valid_BTC] = delayed_audio_BTC[b_idx_BTC[valid_BTC], new_t_BTC[valid_BTC], c_idx_BTC[valid_BTC]]
    return result_BTC
