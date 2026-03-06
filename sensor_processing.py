import numpy as np

def downsample_lidar(ranges, target_beams=60, max_range=3.5):
    """
    Downsample and normalize a LiDAR scan to a fixed number of beams.

    NaN values are replaced with max_range, positive infinity is clipped to
    max_range, and negative infinity is clipped to 0.0. The output is
    normalized to the range [0, 1].

    Args:
        ranges: Iterable of raw LiDAR range values.
        target_beams (int): Number of output beams after downsampling.
        max_range (float): Maximum LiDAR range used for clipping and normalization.

    Returns:
        np.ndarray: Float32 array of shape (target_beams,) with normalized values.
    """
    arr = np.array(ranges, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=max_range, posinf=max_range, neginf=0.0)
    arr = np.clip(arr, 0.0, max_range)

    if len(arr) == 0:
        return np.zeros(target_beams, dtype=np.float32)

    idx = np.linspace(0, len(arr) - 1, target_beams).astype(int)
    ds = arr[idx] / max_range
    return ds.astype(np.float32)
