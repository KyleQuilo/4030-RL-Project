import numpy as np

def downsample_lidar(ranges, target_beams=60, max_range=3.5):
    """
    Downsample a LaserScan ranges array to a fixed number of beams and normalize.
    Args:
        ranges: iterable of float scan ranges
        target_beams: number of beams to output
        max_range: max lidar range for clipping and normalization
    Returns:
        np.ndarray shape (target_beams,) float32 in [0, 1]
    """
    arr = np.array(ranges, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=max_range, posinf=max_range, neginf=0.0)
    arr = np.clip(arr, 0.0, max_range)

    idx = np.linspace(0, len(arr) - 1, target_beams).astype(int)
    ds = arr[idx] / max_range
    return ds.astype(np.float32)
