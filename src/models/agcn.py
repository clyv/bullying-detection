"""Adaptive ST-GCN: learnable graph topology + multi-scale temporal convolution.

The Phase 1 baseline (models/stgcn.py) fixes its adjacency to the COCO-17 skeleton
and convolves time with a single 9x1 kernel. Both are 2018-era choices and both hurt
this task specifically:

* **Fixed topology.** Aggression is defined by relations the skeleton graph does not
  contain — a fist reaching the other person's head is two joints with no edge
  between them. Adding a learnable topology term (and a data-dependent one) lets the
  network form those edges, which is the core idea behind 2s-AGCN and its successors.
* **Single temporal scale.** A shove lasts a few frames; a sustained confrontation
  lasts seconds. One dilation cannot see both, so the temporal branch here is
  multi-scale (dilations 1 and 2, plus a max-pool and a bottleneck path).

Kept deliberately drop-in: same ``(N, C, T, V, M)`` input contract, same constructor
signature as STGCNBaseline, so configs can switch with ``model.name``.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.graph import Graph


class AdaptiveGraphConv(nn.Module):
    """Spatial graph conv whose adjacency is ``A_fixed + B_learned + C_data``.

    ``A_fixed`` is the skeleton's own partitioned adjacency. ``B`` starts as a copy
    of it and is free to grow edges the anatomy does not have. ``C`` is computed per
    sample from the features themselves, so the graph can differ between a shove and
    a handshake in the same batch.
    """

    def __init__(self, in_channels, out_channels, A, embed_ratio=4, adaptive=True):
        super().__init__()
        self.num_partitions = A.size(0)
        self.adaptive = adaptive
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_partitions, kernel_size=1)
        self.register_buffer("A_fixed", A.clone())
        self.B = nn.Parameter(A.clone())

        inner = max(out_channels // embed_ratio, 8)
        self.inner = inner
        self.theta = nn.Conv2d(in_channels, inner * self.num_partitions, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, inner * self.num_partitions, kernel_size=1)
        # Starts at zero so the data-dependent term contributes nothing at init and
        # the block begins life as a plain (fixed + learnable) graph conv.
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """(N', C, T, V) -> (N', out_channels, T, V), N' being batch*persons."""
        n, _, t, v = x.size()
        k = self.num_partitions

        A = self.A_fixed + self.B
        if self.adaptive:
            theta = self.theta(x).view(n, k, self.inner, t, v).mean(dim=3)  # (n, k, inner, v)
            phi = self.phi(x).view(n, k, self.inner, t, v).mean(dim=3)
            # (n, k, v, v) similarity between every joint pair, per partition.
            attention = torch.tanh(torch.einsum("nkcv,nkcw->nkvw", theta, phi) / self.inner)
            A = A.unsqueeze(0) + self.alpha * attention
        else:
            A = A.unsqueeze(0).expand(n, -1, -1, -1)

        feats = self.conv(x).view(n, k, -1, t, v)
        out = torch.einsum("nkctv,nkvw->nctw", feats, A)
        return self.bn(out.contiguous())


class MultiScaleTCN(nn.Module):
    """Temporal convolution over several receptive fields at once.

    Four parallel branches — dilation 1, dilation 2, max-pool, and a 1x1 bottleneck —
    concatenated back to ``out_channels``. The short branches catch impacts, the
    dilated ones catch the build-up and aftermath that distinguish a fight from a
    single frame of contact.
    """

    def __init__(self, channels, out_channels, kernel_size=5, stride=1, dilations=(1, 2)):
        super().__init__()
        num_branches = len(dilations) + 2
        branch_channels = out_channels // num_branches
        # The bottleneck branch absorbs any remainder so the concat is exact.
        last_channels = out_channels - branch_channels * (num_branches - 1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            padding = (kernel_size - 1) * dilation // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, kernel_size=1),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        branch_channels,
                        branch_channels,
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=(padding, 0),
                        dilation=(dilation, 1),
                    ),
                    nn.BatchNorm2d(branch_channels),
                )
            )
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(channels, branch_channels, kernel_size=1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(branch_channels),
            )
        )
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(channels, last_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(last_channels),
            )
        )

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class AGCNBlock(nn.Module):
    """Adaptive spatial graph conv followed by a multi-scale temporal conv."""

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super().__init__()
        self.gcn = AdaptiveGraphConv(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn = MultiScaleTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = 0 if self.residual is None else self.residual(x)
        out = self.tcn(self.relu(self.gcn(x)))
        return self.relu(out + res)


class AGCN(nn.Module):
    """Adaptive ST-GCN. Drop-in replacement for STGCNBaseline.

    Unlike the baseline, the person axis is folded into the batch once at the top and
    unfolded once at the end, so no block has to reason about a 5-D tensor. Pooling
    averages over time and joints, then over people — averaging over people last means
    a single aggressive participant is not diluted by bystanders as much as it would
    be if all three axes were pooled together.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=2,
        num_persons=2,
        graph_strategy="spatial",
        dropout=0.3,
        base_channels=64,
        adaptive=True,
    ):
        super().__init__()
        self.graph = Graph(strategy=graph_strategy)
        A = self.graph.A
        num_node = self.graph.num_node

        self.data_bn = nn.BatchNorm1d(in_channels * num_node * num_persons)

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.blocks = nn.ModuleList(
            [
                AGCNBlock(in_channels, c1, A, residual=False, adaptive=adaptive),
                AGCNBlock(c1, c1, A, adaptive=adaptive),
                AGCNBlock(c1, c1, A, adaptive=adaptive),
                AGCNBlock(c1, c2, A, stride=2, adaptive=adaptive),
                AGCNBlock(c2, c2, A, adaptive=adaptive),
                AGCNBlock(c2, c3, A, stride=2, adaptive=adaptive),
                AGCNBlock(c3, c3, A, adaptive=adaptive),
            ]
        )
        self.dropout = nn.Dropout(dropout)
        # PyTorch's default Linear init, deliberately. 2s-AGCN's std=sqrt(2/num_classes)
        # was tuned for 60 classes (std 0.18); at 2 classes it is std 1.0, which over 256
        # features put 64% of real clips at P > 0.99 before the first step. The pooled
        # run then spent its opening epochs escaping that saturation and sat at chance.
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x):
        n, c, t, v, m = x.size()

        x = x.permute(0, 1, 3, 4, 2).contiguous().view(n, c * v * m, t)
        x = self.data_bn(x)
        x = x.view(n, c, v, m, t).permute(0, 3, 1, 4, 2).contiguous().view(n * m, c, t, v)

        for block in self.blocks:
            x = block(x)

        x = x.view(n, m, x.size(1), x.size(2), x.size(3))
        x = x.mean(dim=(3, 4))  # pool time and joints -> (n, m, channels)
        x = x.mean(dim=1)  # then people -> (n, channels)
        return self.fc(self.dropout(x))
