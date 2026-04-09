"""
Hyperparameters for the DTW Wave Labs sandbox.
Tune these values to adjust sensitivity and window coverage.
"""

# Fibonacci-based sliding window sizes (in candles)
WINDOW_SIZES: list[int] = [21, 55, 89, 144, 233]

# Maximum allowed normalized DTW distance (0.0 = perfect match, 1.0 = worst)
# Tightened to 0.08 — rejects noisy shapes, keeps only clean structural matches.
DTW_DISTANCE_THRESHOLD: float = 0.08

# Number of interpolation points in each ideal template vector
TEMPLATE_RESOLUTION: int = 100

# Savitzky-Golay filter: window_length is set dynamically per window,
# but polyorder stays fixed.
SAVGOL_POLYORDER: int = 2

# Minimum overlap fraction for NMS deduplication (0.5 = 50%)
NMS_OVERLAP_THRESHOLD: float = 0.50

