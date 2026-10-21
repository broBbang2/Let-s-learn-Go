"""급수 조건부 수 예측 신경망 — Maia-2와 같은 아이디어의 작은 버전.

입력: 국면 평면(features.encode) + 급수  →  출력: 그 급수의 사람이 둘 수의 확률 분포 (size*size + 패스)

같은 모델이 두 가지로 쓰인다.
- 실력 추정: 급수 12개를 다 넣어 p(실제 둔 수 | 급수)를 구하고 rank_estimate의 베이즈 갱신에 사용
- 사람다운 수 / 힌트: "추정 급수 + α"를 넣어 나온 분포
그래서 NetPolicy는 KataGoEngine.human_policies와 같은 모양으로 결과를 돌려준다(갈아 끼울 수 있게).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import PLANES, encode, index_vertex
from .rank_estimate import RANKS


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class ResBlock(nn.Module):
    def __init__(self, ch, n_ranks):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        # 블록마다 급수 정보를 채널별 편향으로 다시 넣는다. 처음 한 번만 넣으면 깊은 층에서 희미해진다
        self.rank_bias = nn.Embedding(n_ranks, ch)

    def forward(self, x, rank):
        h = F.relu(self.bn1(self.conv1(x)) + self.rank_bias(rank)[:, :, None, None])
        return F.relu(x + self.bn2(self.conv2(h)))


class RankPolicyNet(nn.Module):
    def __init__(self, planes=PLANES, channels=64, blocks=6, n_ranks=len(RANKS)):
        super().__init__()
        self.config = {"planes": planes, "channels": channels, "blocks": blocks, "n_ranks": n_ranks}
        # +1: 전부 1인 평면. 합성곱의 0 패딩과 구분돼 판 가장자리를 알 수 있다
        self.stem = nn.Sequential(nn.Conv2d(planes + 1, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.ModuleList(ResBlock(channels, n_ranks) for _ in range(blocks))
        self.policy = nn.Conv2d(channels, 1, 1)
        self.pass_head = nn.Linear(channels, 1)

    def forward(self, x, rank):
        """x: (B, planes, n, n) float, rank: (B,) long → logits (B, n*n + 1). 둘 수 없는 점은 -inf."""
        h = self.stem(torch.cat([x, torch.ones_like(x[:, :1])], 1))
        for block in self.blocks:
            h = block(h, rank)
        board = self.policy(h).flatten(1)  # (row, col) 순서 = features.move_index
        logits = torch.cat([board, self.pass_head(h.mean((2, 3)))], 1)
        # 돌이 있는 점(평면 0·1)과 패로 금지된 점(평면 11)은 둘 수 없다
        illegal = torch.cat([((x[:, 0] + x[:, 1] + x[:, 11]) > 0).flatten(1), torch.zeros_like(logits[:, :1], dtype=torch.bool)], 1)
        return logits.masked_fill(illegal, float("-inf"))


def save(model, path, extra=None):
    torch.save({"config": model.config, "state": model.state_dict(), **(extra or {})}, path)


def load(path, device=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = RankPolicyNet(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    return model.to(device or pick_device()).eval(), ckpt


class NetPolicy:
    """학습한 모델을 KataGoEngine.human_policies와 같은 인터페이스로 감싼다."""

    def __init__(self, path, device=None):
        self.device = device or pick_device()
        self.model, self.ckpt = load(path, self.device)

    @torch.no_grad()
    def human_policies(self, board, profiles):
        x = torch.from_numpy(encode(board)).float()[None].to(self.device)
        ranks = [RANKS.index(p.removeprefix("rank_")) for p in profiles]
        logits = self.model(x.expand(len(ranks), -1, -1, -1), torch.tensor(ranks, device=self.device))
        probs = torch.softmax(logits, 1).cpu().numpy()
        n = board.size
        out = {}
        for p, row in zip(profiles, probs):
            nz = np.nonzero(row > 1e-6)[0]
            out[p] = {index_vertex(int(i), n, board): float(row[i]) for i in nz}
        return out
