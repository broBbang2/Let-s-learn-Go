"""급수 조건부 수 예측 모델 학습 (PyTorch, 맥 GPU = mps).

python -m adaptive_go.train data/train/<생성 시각> --epochs 8

- 손실: teacher(KataGo human 모델) 분포와의 교차 엔트로피 = 지식 증류
- 데이터 증강: 배치마다 8가지 대칭(회전·뒤집기) 중 하나 → 바둑판은 돌려도 같은 국면
- 검증: 게임 번호 해시로 고른 10% 판은 학습에 안 쓰고 지표만 잰다 (is_val)
  · loss: teacher 분포 대비 교차 엔트로피 (낮을수록 teacher를 잘 흉내)
  · top1_played: 실제로 둔 수를 1순위로 맞힌 비율
  · top1_teacher: teacher가 가장 높게 본 수와 1순위가 같은 비율
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import progress
from .features import PLANES, symmetry_index_table, symmetry_planes
from .model import RankPolicyNet, pick_device, save

ROOT = Path(__file__).resolve().parent.parent


def is_val(game):
    """게임 번호 → 검증용 여부(약 10%).

    번호를 그대로 쓰면 datagen의 번호 규칙(라운드×12+급수)과 겹쳐 특정 급수만 검증에 들어간다
    ("끝자리 0"은 짝수 급수만, 곱셈 해시도 12의 배수 간격 때문에 급수별로 크게 치우쳤다).
    그래서 표준 해시(MD5)로 번호를 섞는다. 판 수가 수천 개라 판마다 한 번 계산해도 충분히 빠르다.
    """
    games = np.asarray(game)
    flags = {int(g): hashlib.md5(str(int(g)).encode()).digest()[0] % 10 == 0 for g in np.unique(games)}
    return np.vectorize(flags.__getitem__, otypes=[bool])(games) if games.ndim else flags[int(games)]


def load_shards(dirs):
    parts = [np.load(f) for d in dirs for f in sorted(Path(d).glob("shard-*.npz"))]
    if not parts:
        raise SystemExit(f"npz 조각이 없음: {dirs}")
    keys = ["planes", "top_idx", "top_p", "played", "rank", "game", "move_no"]
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    data["size"] = int(parts[0]["size"])
    return data


class Batches:
    def __init__(self, data, indices, batch_size, augment, rng):
        self.d, self.idx, self.bs, self.augment, self.rng = data, indices, batch_size, augment, rng
        self.n = data["size"]
        self.table = symmetry_index_table(self.n)

    def __iter__(self):
        order = self.rng.permutation(self.idx) if self.augment else self.idx
        for s in range(0, len(order), self.bs):
            yield self.make(order[s : s + self.bs])

    def __len__(self):
        return math.ceil(len(self.idx) / self.bs)

    def make(self, b):
        d, n = self.d, self.n
        x = np.unpackbits(d["planes"][b], axis=1)[:, : PLANES * n * n].reshape(len(b), PLANES, n, n)
        top_idx, top_p, played = d["top_idx"][b].astype(np.int64), d["top_p"][b].astype(np.float32), d["played"][b].astype(np.int64)
        if self.augment:
            k = int(self.rng.integers(8))
            x = symmetry_planes(x, k)
            valid = top_idx >= 0
            top_idx = np.where(valid, self.table[k][np.where(valid, top_idx, 0)], -1)
            played = self.table[k][played]
        target = np.zeros((len(b), n * n + 1), dtype=np.float32)
        rows = np.repeat(np.arange(len(b)), top_idx.shape[1])
        keep = top_idx.reshape(-1) >= 0
        target[rows[keep], top_idx.reshape(-1)[keep]] = top_p.reshape(-1)[keep]
        target /= target.sum(1, keepdims=True)  # 상위 K개만 저장했으니 합이 1이 되게
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(d["rank"][b].astype(np.int64)),
            torch.from_numpy(target),
            torch.from_numpy(played),
        )


def run_epoch(model, batches, device, opt=None, sched=None):
    train = opt is not None
    model.train(train)
    tot = {"loss": 0.0, "top1_played": 0.0, "top1_teacher": 0.0, "n": 0}
    with torch.set_grad_enabled(train):
        for x, rank, target, played in batches:
            x, rank, target, played = x.to(device), rank.to(device), target.to(device), played.to(device)
            logits = model(x, rank)
            logp = torch.log_softmax(logits, 1)
            # -inf(둘 수 없는 점) × 0 = nan 방지: target이 0인 곳은 빼고 더한다
            loss = -(torch.where(target > 0, target * logp, torch.zeros_like(logp))).sum(1).mean()
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            pred = logits.argmax(1)
            m = len(x)
            tot["loss"] += loss.item() * m
            tot["top1_played"] += (pred == played).float().sum().item()
            tot["top1_teacher"] += (pred == target.argmax(1)).float().sum().item()
            tot["n"] += m
    n = tot.pop("n")
    return {k: round(v / n, 4) for k, v in tot.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description="급수 조건부 수 예측 모델 학습")
    p.add_argument("data", nargs="+", help="datagen 출력 폴더(들)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--out", default=str(ROOT / "models/rank_policy.pt"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    progress.tee_stdout("train")
    started = time.time()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = load_shards(args.data)
    val = is_val(data["game"])
    train_idx, val_idx = np.nonzero(~val)[0], np.nonzero(val)[0]
    print(f"국면 학습 {len(train_idx)} / 검증 {len(val_idx)}, 판 크기 {data['size']}", flush=True)

    device = pick_device()
    model = RankPolicyNet(channels=args.channels, blocks=args.blocks).to(device)
    print(f"파라미터 {sum(p.numel() for p in model.parameters()):,}개, 장치 {device}", flush=True)
    train_batches = Batches(data, train_idx, args.batch, True, rng)
    val_batches = Batches(data, val_idx, 512, False, rng)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_batches))

    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        tr = run_epoch(model, train_batches, device, opt, sched)
        va = run_epoch(model, val_batches, device)
        history.append({"epoch": epoch, "train": tr, "val": va, "seconds": round(time.time() - t)})
        print(f"[{epoch}/{args.epochs}] epoch {epoch}: train {tr} | val {va} | {time.time() - t:.0f}초"
              f"   {progress.status(epoch, args.epochs, started)}", flush=True)
        if best is None or va["loss"] < best:
            best = va["loss"]
            save(model.cpu(), args.out, {"history": history, "data": args.data})
            model.to(device)
    Path(args.out).with_suffix(".json").write_text(json.dumps(history, indent=2))
    print(f"저장: {args.out} (최저 검증 손실 {best})")


if __name__ == "__main__":
    main()
