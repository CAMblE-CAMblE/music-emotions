"""Music generation package based on conditional VAE."""

LABEL_TO_IDX = {
    "Happy": 0,
    "Angry": 1,
    "Sad": 2,
    "Relaxed": 3,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
