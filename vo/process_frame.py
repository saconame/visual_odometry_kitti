from vo.update_candidates import (
    replenish_candidates,
    triangulate_candidates,
    update_candidates,
)
from vo.rebootstrap import rebootstrap_candidates
from vo.update_landmarks import update_landmarks


def process_frame(I_i, I_i_minus_1, S_i_minus_1):
    state = update_landmarks(I_i_minus_1, I_i, S_i_minus_1)
    state = update_candidates(I_i_minus_1, I_i, state)
    state = triangulate_candidates(state)
    state = rebootstrap_candidates(I_i, state)
    state = replenish_candidates(I_i, state)
    return state
