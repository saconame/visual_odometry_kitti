import numpy as np


def draw_landmark_count_panel(ax, landmark_count):
    ax.clear()

    x_axis = np.arange(len(landmark_count), dtype=np.int32)
    ax.plot(x_axis, landmark_count, linewidth=2.0, color="#2ca02c")

    ax.set_title("Landmark Count")
    ax.set_xlabel("frame")
    ax.set_ylabel("count")
    ax.grid(True, linewidth=0.4, alpha=0.6)
