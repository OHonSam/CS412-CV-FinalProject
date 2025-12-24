from typing import Tuple
import torch
import torch.nn as nn

# Prefer torchvision ResNet (matches the paper's CNN feature extraction idea),
# but provide a pure-PyTorch fallback if torchvision cannot be imported.
try:
    import torchvision
    _HAS_TORCHVISION = True
except Exception:
    torchvision = None
    _HAS_TORCHVISION = False


class SimpleCNN(nn.Module):
    """A small CNN fallback (no pretrained weights)."""
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x).flatten(1)
        return self.proj(feats)


class CNNEncoder(nn.Module):
    """Pretrained ResNet encoder returning a per-frame feature vector."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, out_dim: int = 512, trainable: bool = False):
        super().__init__()
        self.out_dim = out_dim

        if not _HAS_TORCHVISION:
            self.fallback = SimpleCNN(out_dim=out_dim)
            return

        if backbone == "resnet18":
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif backbone == "resnet50":
            m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.features = nn.Sequential(*list(m.children())[:-1])  # avgpool output (B,feat,1,1)
        self.proj = nn.Identity()
        if feat_dim != out_dim:
            self.proj = nn.Linear(feat_dim, out_dim)
        if not trainable:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T,3,H,W)
        if not _HAS_TORCHVISION:
            return self.fallback(x)
        feats = self.features(x).flatten(1)
        feats = self.proj(feats)
        return feats


class TextEncoder(nn.Module):
    """BiLSTM encoder for a token sequence."""
    def __init__(self, vocab_size: int, emb_dim: int = 300, hidden: int = 150, num_layers: int = 1, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.emb(ids))
        out, (h, c) = self.lstm(x)
        h_fwd = h[-2]
        h_bwd = h[-1]
        return torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2*hidden)


class VideoEncoder(nn.Module):
    """LSTM over per-frame CNN features."""
    def __init__(self, in_dim: int, hidden: int = 300, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        out, (h, c) = self.lstm(self.dropout(feats))
        return h[-1]  # (B, hidden)


class CNNLSTM_MCQ4(nn.Module):
    """CNN+LSTM baseline for SUTD in 4-choice setting.

    This mirrors the paper's LSTM-style baselines:
    - encode video frames with CNN then LSTM
    - encode question and each candidate answer with BiLSTM
    - score each (video, question, answer) triple with an MLP and softmax over options
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        cnn_backbone: str = "resnet18",
        cnn_out_dim: int = 512,
        video_hidden: int = 300,
        text_hidden: int = 150,
        emb_dim: int = 300,
        mlp_hidden: int = 512,
        freeze_cnn: bool = True,
    ):
        super().__init__()
        self.cnn = CNNEncoder(backbone=cnn_backbone, out_dim=cnn_out_dim, trainable=not freeze_cnn)
        self.video_lstm = VideoEncoder(in_dim=cnn_out_dim, hidden=video_hidden)
        self.q_enc = TextEncoder(vocab_size=vocab_size, emb_dim=emb_dim, hidden=text_hidden, pad_id=pad_id)
        self.a_enc = TextEncoder(vocab_size=vocab_size, emb_dim=emb_dim, hidden=text_hidden, pad_id=pad_id)

        fused_dim = video_hidden + (2 * text_hidden) * 2
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, frames: torch.Tensor, q_ids: torch.Tensor, a_ids: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        frame_feats = self.cnn(x).view(B, T, -1)
        v = self.video_lstm(frame_feats)

        q = self.q_enc(q_ids)
        a = self.a_enc(a_ids.view(B * 4, -1)).view(B, 4, -1)

        v_rep = v.unsqueeze(1).expand(-1, 4, -1)
        q_rep = q.unsqueeze(1).expand(-1, 4, -1)
        fused = torch.cat([v_rep, q_rep, a], dim=-1)

        return self.mlp(fused).squeeze(-1)  # (B,4)
