# modified from 'https://github.com/amanwalia123/KeyFramesExtraction'
import operator
from collections import Counter

import cv2
import numpy as np
from scipy.signal import argrelextrema

# fixed threshold value
THRESH = 0.2
SMOOTH_WINDOW_SIZE = 3


def pil_to_yuv(pil_image):
    rgb_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    yuv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2YUV)

    return yuv_frame


class FrameData:
    def __init__(self, id, frame, value):
        self.id = id
        self.frame = frame
        self.value = value

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)


def cal_attr(frames):
    new_frames, frames_diff = [], []

    for i in range(1, len(frames)):
        diff = cv2.absdiff(pil_to_yuv(frames[i]), pil_to_yuv(frames[i - 1]))
        count = np.sum(diff)

        frames_diff.append(count)
        new_frame = FrameData(i, pil_to_yuv(frames[i]), count)
        new_frames.append(new_frame)

    frames_diff.insert(0, 0)
    new_frames.insert(0, FrameData(0, pil_to_yuv(frames[0]), 0))

    return new_frames, frames_diff


def rel_change(a, b):
    x = (b - a) / max(a, b) if max(a, b) != 0 else 0.0
    return x


def using_top_order(new_frames, top_n):
    # sort the list in descending order
    new_frames.sort(key=operator.attrgetter("value"), reverse=True)

    return [i.id for i in new_frames[:top_n]]


def using_threshold(new_frames, thresh=THRESH):
    idx_list = []
    for i in range(1, len(new_frames)):
        if (
            abs(
                rel_change(
                    np.float64(new_frames[i - 1].value), np.float64(new_frames[i].value)
                )
            )
            >= thresh
        ):
            idx_list.append(new_frames[i].id)

    return idx_list


def smooth(x, window_len=13, window="hanning"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]

    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = getattr(np, window)(window_len)

    y = np.convolve(w / w.sum(), s, mode="same")

    return y[window_len - 1 : -window_len + 1]


def using_local_maxima(frame_diffs, len_window=SMOOTH_WINDOW_SIZE):
    diff_array = np.array(frame_diffs)
    sm_diff_array = smooth(diff_array, len_window)
    frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]

    return frame_indexes


def get_final_idx_list(top_order_idx_list, threshold_idx_list, local_maxima_idx_list):
    idx_list = []

    sum_array = np.concatenate(
        (
            np.array(top_order_idx_list),
            np.array(threshold_idx_list),
            np.array(local_maxima_idx_list),
        )
    )

    count = Counter(sum_array)
    idx_count_pairs = count.most_common()

    for idx, _ in idx_count_pairs:
        idx_list.append(int(idx))

    return idx_list


def select_frames(frames, top_n):
    """Return exactly *n* frames from *frames* (or fewer if
    *frames* has fewer than *n* elements).  Returns an empty list when *n* <= 0."""
    if not frames or top_n <= 0:
        return []
    if len(frames) <= top_n:
        return frames

    # calculate the relative change between consecutive frames
    new_frames, frames_diff = cal_attr(frames)

    # using top order
    top_order_idx_list = using_top_order(new_frames, top_n)

    # using threshold
    threshold_idx_list = using_threshold(new_frames)

    # using local maxima
    local_maxima_idx_list = using_local_maxima(frames_diff)

    final_idx_list = get_final_idx_list(
        top_order_idx_list, threshold_idx_list, local_maxima_idx_list
    )

    return [frames[i] for i in final_idx_list[:top_n]]
