import numpy as np

from vo.state import init_state
from vision.two_view_init import two_view_initialization


INIT_FRAME_A_INDEX = 0
INIT_FRAME_B_INDEX = 3


def initialize_vo(frames, K, gt_poses):
    frame_a_index = int(INIT_FRAME_A_INDEX)
    frame_b_index = int(INIT_FRAME_B_INDEX)

    gt_t0 = gt_poses[frame_a_index][:3, 3]
    gt_t1 = gt_poses[frame_b_index][:3, 3]
    true_scale = float(np.linalg.norm(gt_t1 - gt_t0))

    init_result = two_view_initialization(
        frames[frame_a_index],
        frames[frame_b_index],
        K,
        scale=true_scale,
    )

    state = init_state()
    state["P"] = init_result["P"]
    state["X"] = init_result["X"]
    state["T_wc"] = init_result["T_wc"]
    state["K"] = K

    init_info = {
        "frame_a_index": frame_a_index,
        "frame_b_index": frame_b_index,
        "stats": init_result["stats"],
    }

    return state, init_info
