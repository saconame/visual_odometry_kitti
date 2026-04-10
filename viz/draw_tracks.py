def draw_frame_panel(ax, image, frame_idx, state):
    ax.clear()
    ax.imshow(image, cmap="gray")
    
    # Add these two lines to lock the frame to the image bounds
    h, w = image.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    if state["P"].size > 0:
        ax.scatter(state["P"][0], state["P"][1], s=6, c="#32cd32", label="P")

    if state["C"].size > 0:
        ax.scatter(state["C"][0], state["C"][1], s=6, c="#ff0000", label="C")

    ax.set_title(f"Frame {frame_idx}")
    ax.set_axis_off()

    if state["P"].size > 0 or state["C"].size > 0:
        ax.legend(loc="lower right", fontsize=8)
