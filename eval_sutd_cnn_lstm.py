import os
import argparse
from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.sutd_lstm_dataset import SUTDMCQ4Dataset
from src.text_tokenizer import Vocab
from src.models import CNNLSTM_MCQ4

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sutd_root", type=str, default="SUTD")
    ap.add_argument("--test_file", type=str, default="questions/R3_test.jsonl")
    ap.add_argument("--videos_dir", type=str, default="videos")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to best.pt or last.pt")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--out_csv", type=str, default="sutd_cnn_lstm_predictions.csv")
    ap.add_argument("--max_test_samples", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    vocab = Vocab.load(ckpt["vocab_path"])

    model = CNNLSTM_MCQ4(
        vocab_size=len(vocab.itos),
        pad_id=vocab.pad_id,
        cnn_backbone=ckpt["args"].get("cnn_backbone", "resnet18"),
        freeze_cnn=ckpt["args"].get("freeze_cnn", True),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    sutd_root = Path(args.sutd_root)
    test_ds = SUTDMCQ4Dataset(
        questions_path=str(sutd_root / args.test_file),
        videos_dir=str(sutd_root / args.videos_dir),
        vocab=vocab,
        num_frames=args.num_frames,
        train=False,
        use_train_aug=False,
        max_samples=args.max_test_samples,
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    rows = []
    correct = 0
    total = 0
    valid_total = 0

    for batch in tqdm(loader, desc="evaluating"):
        valid = batch["valid"].to(device)
        mask = valid == 1
        # run model on all (including invalid) to produce output, but skip invalid for accuracy.
        frames = batch["frames"].to(device)
        q_ids = batch["q_ids"].to(device)
        a_ids = batch["a_ids"].to(device)

        logits = model(frames, q_ids, a_ids)
        preds = logits.argmax(dim=-1).detach().cpu().tolist()

        for rid, fn, pred in zip(batch["record_id"], batch["vid_filename"], preds):
            rows.append((str(rid), str(fn), int(pred)))

        if mask.sum() > 0:
            labels = batch["label"].to(device)[mask]
            preds_t = torch.tensor(preds, device=device)[mask]
            correct += (preds_t == labels).sum().item()
            total += labels.numel()
            valid_total += mask.sum().item()

    acc = correct / max(total, 1)
    print(f"Test accuracy (only samples with video found): {acc:.4f} (valid_samples={valid_total})")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "video", "prediction"])
        w.writerows(rows)
    print(f"Wrote predictions to {out_csv}")

if __name__ == "__main__":
    main()
