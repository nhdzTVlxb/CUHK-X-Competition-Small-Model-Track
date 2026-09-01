from __future__ import annotations

import torch
from torch import nn


class TemporalBranch(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=1),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is (batch, time, features); Conv1d wants (batch, features, time).
        x = x.transpose(1, 2)
        x = self.net(x)
        return torch.amax(x, dim=-1)


class SkelImuNet(nn.Module):
    def __init__(
        self,
        skeleton_dim: int = 68,
        imu_dim: int = 19,
        num_classes: int = 40,
        width: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.skeleton_branch = TemporalBranch(skeleton_dim, width, width, dropout)
        self.imu_branch = TemporalBranch(imu_dim, width, width, dropout)
        self.gate = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.SiLU(),
            nn.Linear(width, 2),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, num_classes),
        )

    def forward(self, skeleton: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        sk = self.skeleton_branch(skeleton)
        im = self.imu_branch(imu)
        merged = torch.cat([sk, im], dim=1)
        gates = self.gate(merged)
        merged = torch.cat([sk * gates[:, :1], im * gates[:, 1:]], dim=1)
        return self.head(merged)


class FrameCnnBranch(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, base: int = 32, dropout: float = 0.15) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU(),
            nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(base * 2, base * 3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 3),
            nn.SiLU(),
            nn.Conv2d(base * 3, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.cnn(x).flatten(1)
        return x.reshape(b, t, -1)


class ResidualTCN(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(width),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class SequenceBranch(nn.Module):
    def __init__(self, in_dim: int, width: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(in_dim, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(
            ResidualTCN(width, 1, dropout),
            ResidualTCN(width, 2, dropout),
            ResidualTCN(width, 4, dropout),
        )
        self.gru = nn.GRU(width, width // 2, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.out = nn.Sequential(nn.LayerNorm(width * 3), nn.Linear(width * 3, width), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x.transpose(1, 2))
        x = self.tcn(x).transpose(1, 2)
        x, _ = self.gru(x)
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1).unsqueeze(-1)
        weighted = (x * weights).sum(dim=1)
        pooled = torch.cat([weighted, x.mean(dim=1), x.amax(dim=1)], dim=1)
        return self.out(pooled)


class FusionNet(nn.Module):
    def __init__(
        self,
        skeleton_dim: int = 68,
        imu_dim: int = 19,
        num_classes: int = 40,
        width: int = 128,
        image_base: int = 32,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.skeleton_branch = TemporalBranch(skeleton_dim, width, width, dropout)
        self.imu_branch = TemporalBranch(imu_dim, width, width, dropout)
        self.depth_frame = FrameCnnBranch(3, width, image_base, dropout=0.10)
        self.ir_frame = FrameCnnBranch(1, width, image_base, dropout=0.10)
        self.depth_temporal = TemporalBranch(width, width, width, dropout)
        self.ir_temporal = TemporalBranch(width, width, width, dropout)
        self.gate = nn.Sequential(
            nn.Linear(width * 4, width * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, 4),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width * 4),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, num_classes),
        )

    def forward(
        self,
        skeleton: torch.Tensor,
        imu: torch.Tensor,
        depth: torch.Tensor,
        ir: torch.Tensor,
    ) -> torch.Tensor:
        sk = self.skeleton_branch(skeleton)
        im = self.imu_branch(imu)
        de = self.depth_temporal(self.depth_frame(depth))
        ir_feat = self.ir_temporal(self.ir_frame(ir))
        merged = torch.cat([sk, im, de, ir_feat], dim=1)
        gates = self.gate(merged)
        gated = torch.cat(
            [
                sk * gates[:, 0:1],
                im * gates[:, 1:2],
                de * gates[:, 2:3],
                ir_feat * gates[:, 3:4],
            ],
            dim=1,
        )
        return self.head(gated)


class FusionNetV2(nn.Module):
    def __init__(
        self,
        skeleton_dim: int = 238,
        imu_dim: int = 19,
        num_classes: int = 40,
        width: int = 160,
        image_base: int = 32,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.skeleton_branch = SequenceBranch(skeleton_dim, width, dropout)
        self.imu_branch = SequenceBranch(imu_dim, width, dropout)
        self.depth_frame = FrameCnnBranch(3, width, image_base, dropout=0.10)
        self.ir_frame = FrameCnnBranch(1, width, image_base, dropout=0.10)
        self.depth_temporal = SequenceBranch(width, width, dropout)
        self.ir_temporal = SequenceBranch(width, width, dropout)
        self.gate = nn.Sequential(
            nn.Linear(width * 4, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, 4),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width * 4),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, num_classes),
        )

    def forward(
        self,
        skeleton: torch.Tensor,
        imu: torch.Tensor,
        depth: torch.Tensor,
        ir: torch.Tensor,
    ) -> torch.Tensor:
        sk = self.skeleton_branch(skeleton)
        im = self.imu_branch(imu)
        de = self.depth_temporal(self.depth_frame(depth))
        ir_feat = self.ir_temporal(self.ir_frame(ir))
        merged = torch.cat([sk, im, de, ir_feat], dim=1)
        gates = self.gate(merged)
        gated = torch.cat(
            [
                sk * gates[:, 0:1],
                im * gates[:, 1:2],
                de * gates[:, 2:3],
                ir_feat * gates[:, 3:4],
            ],
            dim=1,
        )
        return self.head(gated)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
