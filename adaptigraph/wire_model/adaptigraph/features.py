import torch


def turning(P):
    u = P[..., 1:, :] - P[..., :-1, :]
    a, b = u[..., :-1, :], u[..., 1:, :]
    return torch.atan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], (a * b).sum(-1))


def end_actions(P0, P1):
    B, N, _ = P0.shape
    act = torch.zeros(B, N, 3, device=P0.device, dtype=P0.dtype)

    def ang(P, i, j):
        e = P[:, j] - P[:, i]
        return torch.atan2(e[:, 1], e[:, 0])
    for node, (i, j) in ((0, (0, 1)), (N - 1, (N - 2, N - 1))):
        act[:, node, :2] = P1[:, node] - P0[:, node]
        dth = ang(P1, i, j) - ang(P0, i, j)
        act[:, node, 2] = torch.atan2(torch.sin(dth), torch.cos(dth))
    return act
