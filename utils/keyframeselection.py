def select_frames(frames, n):
    """Return exactly *n* uniformly-spaced frames from *frames* (or fewer if
    *frames* has fewer than *n* elements).  Returns an empty list when *n* <= 0."""
    if not frames or n <= 0:
        return []
    if len(frames) <= n:
        return frames
    # Integer arithmetic avoids a numpy allocation for the typical case of n=3.
    indices = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in indices]
