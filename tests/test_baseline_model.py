import numpy as np
import torch

from src.models.graph import Graph
from src.models.stgcn import STGCNBaseline
from src.datasets.unified_loader import UnifiedSkeletonDataset


def test_graph_adjacency_shape():
    graph = Graph(strategy="spatial")
    assert graph.A.shape == (3, 17, 17)
    assert int(graph.A[0].sum()) == 17  # Self-loops identity check


def test_model_forward_pass_dimensions():
    # Shape protocol: (Batch_Size, Channels, Timesteps, Vertices/Joints, Monsters/Actors)
    mock_input = torch.randn(2, 3, 100, 17, 2)
    model = STGCNBaseline(in_channels=3, num_classes=2)
    model.eval()
    with torch.no_grad():
        output = model(mock_input)
    assert output.shape == (2, 2)  # (Batch_size, Num_classes)


def test_loader_padding_and_truncation(tmp_path):
    # Construct short sequence .npz mock file
    short_kp = np.random.randn(20, 2, 17, 2)
    short_scores = np.random.rand(20, 2, 17)
    file_path = tmp_path / "mock_agg_01.npz"
    np.savez(file_path, keypoints=short_kp, scores=short_scores)

    dataset = UnifiedSkeletonDataset(data_dir=str(tmp_path), target_frames=150)
    assert len(dataset) == 1

    tensor_data, label = dataset[0]
    assert tensor_data.shape == (3, 150, 17, 2)  # (C, T, V, M)
    assert label.item() == 1  # Derived correctly from file signature name
