import pathlib
import typing as tp
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

def load_raw_int16(file_path: tf.Tensor) -> tf.Tensor:
    """
    Loads a raw binary file containing int16 data and reshapes it to [-1, 9].

    Args:
      file_path (tf.Tensor): Scalar string tensor representing the file path.
    
    Returns:
      tf.Tensor: An int16 tensor of shape [num_frames, 9].
    """
    raw = tf.io.read_file(file_path)
    audio = tf.io.decode_raw(raw, tf.int16)
    audio = tf.reshape(audio, [-1, 9])
    return audio

def load_pair(text_file: tf.Tensor, audio_file: tf.Tensor) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """
    Loads a pair of files: a text file and its corresponding raw binary audio file.
    
    The text file is read as raw bytes and decoded to a uint8 vector.
    The audio file is read using load_raw_int16.
    
    Args:
      text_file (tf.Tensor): Scalar string tensor representing the text file path.
      audio_file (tf.Tensor): Scalar string tensor representing the audio file path.
    
    Returns:
      A tuple (text, audio) where:
        - text: tf.Tensor of type uint8.
        - audio: tf.Tensor of type int16 with shape [num_frames, 9].
    """
    text = tf.io.decode_raw(tf.io.read_file(text_file), tf.uint8)
    audio = load_raw_int16(audio_file)
    return text, audio

def process_sample(text: tf.Tensor, audio: tf.Tensor, config: DataLoaderConfig) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """
    Pads text and audio tensors to fixed lengths.

    Args:
      text (tf.Tensor): 1D tensor of type uint8.
      audio (tf.Tensor): 2D tensor of type int16 with shape [num_frames, 9].
      config (DataLoaderConfig): Data loader configuration.

    Returns:
      Tuple of padded text and padded audio tensors.
    """
    padded_text = tf.pad(text, [[0, config.text_length - tf.shape(text)[0]]],
                         constant_values=config.pad_value)
    padded_audio = tf.pad(audio, [[0, config.audio_length - tf.shape(audio)[0]], [0, 0]],
                          constant_values=config.pad_value)
    return padded_text, padded_audio

def build_delay_indices(batch_size: int, audio_length: int, delay_pattern: tp.List[int]
                        ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """
    Precompute indices for the delay operation.
    
    Returns:
      A tuple (t_idx_BTC, indices) where:
        - t_idx_BTC is a tensor of shape [B, T, 9] computed as time indices minus the delay.
        - indices is a tensor of shape [B*T*9, 3] used for gathering, computed from:
            batch indices, tf.maximum(t_idx_BTC, 0), and channel indices.
    """
    C = 9
    delay_arr = tf.constant(delay_pattern, dtype=tf.int32)
    # Compute time indices: shape [B, T, 1]
    t_idx_BT1 = tf.broadcast_to(tf.expand_dims(tf.range(audio_length), axis=0), [batch_size, audio_length])
    t_idx_BT1 = tf.expand_dims(t_idx_BT1, axis=-1)
    # Subtract delay per channel: shape becomes [B, T, 9]
    t_idx_BTC = t_idx_BT1 - tf.reshape(delay_arr, [1, 1, C])
    
    # Compute batch and channel indices.
    b_idx_BTC = tf.broadcast_to(tf.reshape(tf.range(batch_size), [batch_size, 1, 1]),
                                [batch_size, audio_length, C])
    c_idx_BTC = tf.broadcast_to(tf.reshape(tf.range(C), [1, 1, C]),
                                [batch_size, audio_length, C])
    
    indices = tf.stack([
        tf.reshape(b_idx_BTC, [-1]),
        tf.reshape(tf.maximum(t_idx_BTC, 0), [-1]),
        tf.reshape(c_idx_BTC, [-1])
    ], axis=1)
    return t_idx_BTC, indices

def build_revert_indices(batch_size: int, audio_length: int, delay_pattern: tp.List[int]
                         ) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """
    Precompute indices for the revert operation.
    
    Returns:
      A tuple (t_idx_BTC, indices) where:
        - t_idx_BTC is a tensor of shape [B, T, 9] computed as time indices plus the delay.
        - indices is a tensor of shape [B*T*9, 3] used for gathering, computed from:
            batch indices, tf.minimum(t_idx_BTC, audio_length - 1), and channel indices.
    """
    C = 9
    delay_arr = tf.constant(delay_pattern, dtype=tf.int32)
    t_idx_BT1 = tf.broadcast_to(tf.expand_dims(tf.range(audio_length), axis=0), [batch_size, audio_length])
    t_idx_BT1 = tf.expand_dims(t_idx_BT1, axis=-1)
    t_idx_BTC = t_idx_BT1 + tf.reshape(delay_arr, [1, 1, C])
    
    b_idx_BTC = tf.broadcast_to(tf.reshape(tf.range(batch_size), [batch_size, 1, 1]),
                                [batch_size, audio_length, C])
    c_idx_BTC = tf.broadcast_to(tf.reshape(tf.range(C), [1, 1, C]),
                                [batch_size, audio_length, C])
    
    indices = tf.stack([
        tf.reshape(b_idx_BTC, [-1]),
        tf.reshape(tf.minimum(t_idx_BTC, audio_length - 1), [-1]),
        tf.reshape(c_idx_BTC, [-1])
    ], axis=1)
    return t_idx_BTC, indices

def apply_audio_delay(audio_BTC: tf.Tensor, pad_value: int, delay_pattern: tp.List[int],
                      precomp: tp.Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """
    Applies a delay pattern to batched audio tokens using precomputed indices.
    
    Args:
      audio_BTC (tf.Tensor): int16 tensor with shape [B, T, 9].
      pad_value (int): Padding value.
      delay_pattern (List[int]): Delay steps for each channel (length 9).
      precomp: Tuple of precomputed delay indices (t_idx_BTC, indices).
    
    Returns:
      tf.Tensor: Delayed audio with shape [B, T, 9], dtype int16.
    """
    t_idx_BTC, indices = precomp
    gathered = tf.reshape(tf.gather_nd(audio_BTC, indices), tf.shape(audio_BTC))
    result = tf.where(t_idx_BTC < 0, tf.cast(pad_value, audio_BTC.dtype), gathered)
    return result

def revert_audio_delay(audio_BTC: tf.Tensor, pad_value: int, delay_pattern: tp.List[int],
                       precomp: tp.Tuple[tf.Tensor, tf.Tensor], T: int) -> tf.Tensor:
    """
    Reverts a delay pattern from batched audio tokens using precomputed indices.
    
    Args:
      audio_BTC (tf.Tensor): int16 tensor with shape [B, T, 9].
      pad_value (int): Padding value.
      delay_pattern (List[int]): Delay pattern (length 9).
      precomp: Tuple of precomputed revert indices (t_idx_BTC, indices).
      T (int): The fixed audio length.
    
    Returns:
      tf.Tensor: Recovered audio with shape [B, T, 9], dtype int16.
    """
    t_idx_BTC, indices = precomp
    gathered = tf.reshape(tf.gather_nd(audio_BTC, indices), tf.shape(audio_BTC))
    result = tf.where(t_idx_BTC >= T, tf.cast(pad_value, audio_BTC.dtype), gathered)
    return result

def map_process_sample(text: tf.Tensor, audio: tf.Tensor, config: DataLoaderConfig) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """
    Mapping function to pad text and audio tensors.
    """
    return process_sample(text, audio, config)

def map_apply_delay(txt: tf.Tensor, audio: tf.Tensor, config: DataLoaderConfig,
                    delay_precomp: tp.Tuple[tf.Tensor, tf.Tensor]) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    """
    Mapping function to apply the audio delay.
    """
    delayed_audio = apply_audio_delay(audio, config.pad_value, config.delay_pattern, delay_precomp)
    return txt, delayed_audio

def create_dataset(data_dir: pathlib.Path, config: DataLoaderConfig, seed: int = 42) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from paired text and audio files.
    
    For each sample, the text file (with extension *.txt) is paired with an audio file
    (with extension *.bin) having the same basename. The text file is read as raw bytes
    and decoded to a uint8 vector, while the audio file is read as a raw binary int16 array
    and reshaped to [T, 9]. Samples are padded, batched, and the audio delay is applied
    using precomputed indices.
    
    Args:
      data_dir (pathlib.Path): Directory containing paired files.
      config (DataLoaderConfig): Data loader configuration.
      seed (int): Shuffle seed.
    
    Returns:
      tf.data.Dataset: Yields tuples (txt_L, delayed_audio_L9) where:
        txt_L: tf.Tensor, shape [batch_size, text_length], dtype tf.uint8.
        delayed_audio_L9: tf.Tensor, shape [batch_size, audio_length, 9], dtype tf.int16.
    """
    # List all .txt files.
    text_files = tf.io.gfile.glob(str(data_dir / "*.txt"))
    text_files.sort()
    # Build corresponding audio file paths by replacing .txt with .bin.
    audio_files = [f.replace('.txt', '.bin') for f in text_files]
    
    ds = tf.data.Dataset.from_tensor_slices((text_files, audio_files))
    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply padding using a named mapping function.
    ds = ds.map(lambda text, audio: map_process_sample(text, audio, config),
                num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(config.batch_size)
    
    # Precompute indices for delay and revert operations.
    delay_precomp = build_delay_indices(config.batch_size, config.audio_length, config.delay_pattern)
    
    # Apply the delay using a named mapping function.
    ds = ds.map(lambda txt, audio: map_apply_delay(txt, audio, config, delay_precomp),
                num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
