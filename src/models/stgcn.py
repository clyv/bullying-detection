import torch
import torch.nn as nn
from src.models.graph import Graph


class SpatialGraphConv(nn.Module):
    """Applies Spatial Graph Convolutions using structural partitions."""

    def __init__(self, in_channels, out_channels, k_size):
        super().__init__()
        self.k_size = k_size
        self.conv = nn.Conv2d(in_channels, out_channels * k_size, kernel_size=1)

    def forward(self, x, A):
        # x shape: (N, C, T, V, M). The 1x1 conv acts per (t, v), so fold the
        # person axis M into the batch to get a 4-D tensor Conv2d accepts.
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        x = self.conv(x)  # -> (N*M, out_channels * k_size, T, V)
        x = x.view(N, M, self.k_size, -1, T, V).permute(0, 2, 3, 4, 5, 1)

        # Aggregate across spatial nodes per adjacency partition.
        output = torch.einsum("nkctvm,kvw->nctwm", x, A.to(x.device))
        return output.contiguous()


class STGCNBlock(nn.Module):
    """A standard Spatio-Temporal block containing spatial GCN and temporal Conv layers."""

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, A.size(0))
        self.A = nn.Parameter(A, requires_grad=False)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
            ),
            nn.BatchNorm2d(out_channels),
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, V, M = x.size()
        # Residual is computed in 4-D (N, C, T, V*M) space; it is either a tensor
        # of the block's output shape or the int 0 (when residual=False).
        res = self.residual(x.view(N, C, T, V * M))

        x = self.gcn(x, self.A)
        x = x.view(N, -1, T, V * M)
        x = self.tcn(x)  # temporal stride may shrink T
        x = x + res
        x = x.view(N, -1, x.size(2), V, M)
        return self.relu(x)


class STGCNBaseline(nn.Module):
    """Full Spatio-Temporal Graph Convolutional Network baseline model."""

    def __init__(
        self, in_channels=3, num_classes=2, num_persons=2, graph_strategy="spatial", dropout=0.3
    ):
        super().__init__()
        self.graph = Graph(strategy=graph_strategy)
        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(in_channels * self.graph.num_node * num_persons)

        self.layer1 = STGCNBlock(in_channels, 64, A, residual=False)
        self.layer2 = STGCNBlock(64, 64, A)
        self.layer3 = STGCNBlock(64, 128, A, stride=2)
        self.layer4 = STGCNBlock(128, 256, A, stride=2)

        self.dropout = nn.Dropout(dropout)  # light regularization before the classifier
        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 1, 3, 4, 2).contiguous().view(N, C * V * M, T)
        x = self.data_bn(x).view(N, C, V, M, T).permute(0, 1, 4, 2, 3).contiguous()

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pool over the remaining (T, V, M) dims, then classify.
        x = nn.functional.avg_pool3d(x, kernel_size=x.size()[2:])  # (N, 256, 1, 1, 1)
        x = self.dropout(x.view(N, -1, 1, 1))  # Conv2d needs 4-D -> (N, 256, 1, 1)
        x = self.fcn(x)  # (N, num_classes, 1, 1)
        return x.view(N, -1)  # (N, num_classes)
