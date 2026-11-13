import numpy as np

BOARD_W_MM = 1200.0
BOARD_H_MM = 1680.0


def board_camera(tf, height_m=2.0, lookat_mm=None):
    M = np.array([tf.e_x, tf.e_y]).T
    Minv = np.linalg.inv(M)
    right = Minv @ np.array([1.0, 0.0])
    up = Minv @ np.array([0.0, -1.0])
    right_3d = np.array([right[0], right[1], 0.0])
    up_3d = np.array([up[0], up[1], 0.0])
    if lookat_mm is None:
        lookat_mm = (BOARD_W_MM / 2.0, BOARD_H_MM / 2.0)
    lookat = np.array([lookat_mm[0] / 1000.0, lookat_mm[1] / 1000.0, 0.0])
    pos = lookat + np.array([0.0, 0.0, height_m])
    return pos, right_3d, up_3d


def camera_xml(tf, render_h_px, name="video_match", lookat_mm=None):
    pos, right, up = board_camera(tf, lookat_mm=lookat_mm)
    fovy_m = render_h_px * tf.mm_per_px / 1000.0
    xml = (f'<camera name="{name}" mode="fixed" projection="orthographic" '
          f'fovy="{fovy_m:.6f}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
          f'xyaxes="{right[0]:.6f} {right[1]:.6f} {right[2]:.6f} '
          f'{up[0]:.6f} {up[1]:.6f} {up[2]:.6f}"/>')
    return xml, fovy_m


def crop_camera(tf, frame_shape, corners_mm=((70, 70), (1130, 70), (1130, 1680), (70, 1680)),
                name="video_match", scale=1.0):
    H, W = frame_shape[:2]
    pxc = tf.to_px(np.array(corners_mm, dtype=float))
    x0, y0 = max(0, int(pxc[:, 0].min())), max(0, int(pxc[:, 1].min()))
    x1, y1 = min(W, int(pxc[:, 0].max())), min(H, int(pxc[:, 1].max()))
    off = np.array([x0, y0])
    rw, rh = x1 - x0, y1 - y0
    centre_px = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    centre_mm = tf.to_mm(centre_px)
    pos, right, up = board_camera(tf, lookat_mm=tuple(centre_mm))
    fovy_m = rh * tf.mm_per_px / 1000.0
    rh_out, rw_out = max(1, round(rh * scale)), max(1, round(rw * scale))
    xml = (f'<camera name="{name}" mode="fixed" projection="orthographic" '
          f'fovy="{fovy_m:.6f}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
          f'xyaxes="{right[0]:.6f} {right[1]:.6f} {right[2]:.6f} '
          f'{up[0]:.6f} {up[1]:.6f} {up[2]:.6f}"/>')
    return xml, rh_out, rw_out, off
