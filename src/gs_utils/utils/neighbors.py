"""Nearest-neighbor helper functions."""

import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


def knn(points: Tensor, num_neighbors: int = 4) -> Tensor:
    """Return Euclidean distances to the nearest neighbors for each point."""
    point_array = points.cpu().numpy()
    nearest_neighbor_model = NearestNeighbors(
        n_neighbors=num_neighbors,
        metric="euclidean",
    ).fit(point_array)
    distances, _ = nearest_neighbor_model.kneighbors(point_array)
    return torch.from_numpy(distances).to(points)
