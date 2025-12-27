import os
import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from src.sutd_lstm_dataset import SUTDMCQ4Dataset
from src.text_tokenizer import build_vocab, Vocab
from src.sutd_io import load_sutd_jsonl, to_mcq4
from src.models import CNNLSTM_MCQ4

def build_or_load_vocab(vocab_path: str, train_questions_path: str, max_size: int = 40000, min_freq: int = 2) -> Vocab:
    if os.path.exists(vocab_path):
        return Vocab.load(vocab_path)
    items = load_sutd_jsonl(train_questions_path)
    texts: List[str] = []
    for it in items:
        _, _, q, choices, _ = to_mcq4(it)
        texts.append(q)
        texts.extend(choices)
    vocab = build_vocab(texts, min_freq=min_freq, max_size=max_size)
    Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
    vocab.save(vocab_path)
    print(f"Saved vocab to {vocab_path} (size={len(vocab.itos)})")
    return vocab

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    valid_total = 0
    for batch in loader:
        valid = batch["valid"].to(device)
        mask = valid == 1
        if mask.sum() == 0:
            continue
        frames = batch["frames"].to(device)[mask]
        q_ids = batch["q_ids"].to(device)[mask]
        a_ids = batch["a_ids"].to(device)[mask]
        labels = batch["label"].to(device)[mask]

        logits = model(frames, q_ids, a_ids)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        valid_total += mask.sum().item()
    acc = correct / max(total, 1)
    return acc, valid_total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sutd_root", type=str, default="SUTD", help="Path to SUTD folder (contains videos/ and questions/)")
    ap.add_argument("--train_file", type=str, default="questions/R3_train.jsonl")
    ap.add_argument("--val_file", type=str, default="questions/R3_val.jsonl")
    ap.add_argument("--videos_dir", type=str, default="videos")
    ap.add_argument("--out_dir", type=str, default="outputs/sutd_cnn_lstm")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--freeze_cnn", action="store_true", default=True)
    ap.add_argument("--cnn_backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--use_train_aug", action="store_true")
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_val_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    sutd_root = Path(args.sutd_root)
    train_q = sutd_root / args.train_file
    val_q = sutd_root / args.val_file
    videos_dir = sutd_root / args.videos_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = str(out_dir / "vocab.json")
    vocab = build_or_load_vocab(vocab_path, str(train_q))

    device = args.device

    train_ds = SUTDMCQ4Dataset(
        questions_path=str(train_q),
        videos_dir=str(videos_dir),
        vocab=vocab,
        num_frames=args.num_frames,
        train=True,
        use_train_aug=args.use_train_aug,
        max_samples=args.max_train_samples,
    )
    val_ds = SUTDMCQ4Dataset(
        questions_path=str(val_q),
        videos_dir=str(videos_dir),
        vocab=vocab,
        num_frames=args.num_frames,
        train=False,
        use_train_aug=False,
        max_samples=args.max_val_samples,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = CNNLSTM_MCQ4(
        vocab_size=len(vocab.itos),
        pad_id=vocab.pad_id,
        cnn_backbone=args.cnn_backbone,
        freeze_cnn=args.freeze_cnn,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        running_loss = 0.0
        n = 0
        for batch in pbar:
            valid = batch["valid"].to(device)
            mask = valid == 1
            if mask.sum() == 0:
                continue
            frames = batch["frames"].to(device)[mask]
            q_ids = batch["q_ids"].to(device)[mask]
            a_ids = batch["a_ids"].to(device)[mask]
            labels = batch["label"].to(device)[mask]

            logits = model(frames, q_ids, a_ids)
            loss = F.cross_entropy(logits, labels)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            pbar.set_postfix(loss=running_loss / max(n, 1))

        val_acc, val_n = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: val_acc={val_acc:.4f} (valid_samples={val_n})")

        ckpt = {
            "model": model.state_dict(),
            "vocab_path": vocab_path,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_acc > best:
            best = val_acc
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  New best: {best:.4f} -> saved best.pt")

    print(f"Done. Best val acc: {best:.4f}")

if __name__ == "__main__":
    main()
