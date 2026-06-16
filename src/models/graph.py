import numpy as np
import torch


class Graph:
    """COCO-17 Layout Graph Topology for Spatial-Temporal GCN Convolutions."""

    def __init__(self, strategy="spatial"):
        self.num_node = 17
        self.strategy = strategy

        # Defining COCO skeletal edge pairs (0: Nose, 1-4: Face, 5-6: Shoulders, etc.)
        self.edges = [
            (15, 13),
            (13, 11),
            (16, 14),
            (14, 12),
            (11, 12),
            (5, 11),
            (6, 12),
            (5, 6),
            (5, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (1, 2),
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
        ]
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        I = np.eye(self.num_node)
        A_binary = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            A_binary[i, j] = 1
            A_binary[j, i] = 1

        if self.strategy == "uniform":
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = A_binary + I
            return torch.tensor(A, dtype=torch.float32)

        elif self.strategy == "spatial":
            # Partition Subsets: 0=Self-loops, 1=Symmetric neighborhood links
            A = np.zeros((3, self.num_node, self.num_node))
            A[0] = I
            A[1] = A_binary
            A[2] = A_binary * 0.5  # Symmetrical extension mapping
            return torch.tensor(A, dtype=torch.float32)

        return torch.tensor(A_binary + I, dtype=torch.float32).unsqueeze(0)