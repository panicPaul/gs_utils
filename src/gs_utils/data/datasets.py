"""Typed dataset helpers built on top of parsed-scene metadata."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from gs_utils.contracts import RenderInput
from gs_utils.data.colmap.parser import ColmapParser
from gs_utils.data.config import DatasetConfig, DataSourceConfig
from gs_utils.data.contract import (
    DataSample,
    DepthNormalSample,
    DepthSample,
    NormalSample,
    ParsedScene,
    PointCloud,
)
from gs_utils.utils.random import get_numpy_rng


def _resize_image(
    image: np.ndarray, downsample_factor: int
) -> tuple[np.ndarray, float, float]:
    if downsample_factor <= 1:
        return image, 1.0, 1.0
    resized_size = (
        max(round(image.shape[1] / downsample_factor), 1),
        max(round(image.shape[0] / downsample_factor), 1),
    )
    resized = np.array(
        Image.fromarray(image).resize(resized_size, Image.BICUBIC)
    )
    scale_y = resized.shape[0] / image.shape[0]
    scale_x = resized.shape[1] / image.shape[1]
    return resized, scale_x, scale_y


def _resize_mask(mask: np.ndarray, downsample_factor: int) -> np.ndarray:
    if downsample_factor <= 1:
        return mask
    resized_size = (
        max(round(mask.shape[1] / downsample_factor), 1),
        max(round(mask.shape[0] / downsample_factor), 1),
    )
    return (
        np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                resized_size, Image.NEAREST
            )
        )
        > 0
    )


def _find_aux_path(image_path: Path, directory: Path) -> Path | None:
    stem = image_path.stem
    for suffix in (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _load_aux_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    return imageio.imread(path)


def _resize_depth(depth: np.ndarray, downsample_factor: int) -> np.ndarray:
    if downsample_factor <= 1:
        return depth
    depth_tensor = torch.from_numpy(depth.astype(np.float32))
    if depth_tensor.ndim == 2:
        depth_tensor = depth_tensor[None, None]
    elif depth_tensor.ndim == 3:
        depth_tensor = depth_tensor.permute(2, 0, 1)[None]
    else:
        raise ValueError(f"Unsupported depth shape: {depth.shape}")
    resized = F.interpolate(
        depth_tensor,
        size=(
            max(round(depth.shape[0] / downsample_factor), 1),
            max(round(depth.shape[1] / downsample_factor), 1),
        ),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0].permute(1, 2, 0).cpu().numpy()


def _resize_normals(normals: np.ndarray, downsample_factor: int) -> np.ndarray:
    if downsample_factor <= 1:
        return normals
    normals_tensor = torch.from_numpy(normals.astype(np.float32))
    if normals_tensor.ndim != 3:
        raise ValueError(f"Unsupported normals shape: {normals.shape}")
    resized = F.interpolate(
        normals_tensor.permute(2, 0, 1)[None],
        size=(
            max(round(normals.shape[0] / downsample_factor), 1),
            max(round(normals.shape[1] / downsample_factor), 1),
        ),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0].permute(1, 2, 0).cpu().numpy()


class ParsedSceneDataset(Dataset[DataSample]):
    """Dataset that materializes typed samples from a parsed scene."""

    def __init__(
        self,
        parsed_scene: ParsedScene,
        indices: Sequence[int] | None = None,
        patch_size: int | None = None,
        downsample_factor: int = 1,
        preload: bool = False,
        depth_dir: Path | None = None,
        normals_dir: Path | None = None,
    ) -> None:
        self.parsed_scene = parsed_scene
        self.indices = (
            list(indices)
            if indices is not None
            else list(range(len(parsed_scene.frames)))
        )
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor
        self.preload = preload
        self.depth_dir = depth_dir
        self.normals_dir = normals_dir
        self._image_cache: dict[int, np.ndarray] | None = None
        self._depth_cache: dict[int, np.ndarray] | None = None
        self._normals_cache: dict[int, np.ndarray] | None = None
        if preload:
            self._image_cache = {
                index: imageio.imread(parsed_scene.frames[index].image_path)[
                    ..., :3
                ]
                for index in self.indices
            }
            if depth_dir is not None:
                self._depth_cache = {}
                for index in self.indices:
                    aux_path = _find_aux_path(
                        parsed_scene.frames[index].image_path, depth_dir
                    )
                    if aux_path is not None:
                        self._depth_cache[index] = _load_aux_array(aux_path)
            if normals_dir is not None:
                self._normals_cache = {}
                for index in self.indices:
                    aux_path = _find_aux_path(
                        parsed_scene.frames[index].image_path, normals_dir
                    )
                    if aux_path is not None:
                        self._normals_cache[index] = _load_aux_array(aux_path)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> DataSample:
        index = self.indices[item]
        frame = self.parsed_scene.frames[index]
        if self._image_cache is not None:
            image = np.array(self._image_cache[index], copy=True)
        else:
            image = imageio.imread(frame.image_path)[..., :3]
        mask = None if frame.mask is None else frame.mask.cpu().numpy()
        depth = None
        normals = None
        if self.depth_dir is not None:
            if self._depth_cache is not None:
                depth = self._depth_cache.get(index)
            else:
                aux_path = _find_aux_path(frame.image_path, self.depth_dir)
                if aux_path is not None:
                    depth = _load_aux_array(aux_path)
        if self.normals_dir is not None:
            if self._normals_cache is not None:
                normals = self._normals_cache.get(index)
            else:
                aux_path = _find_aux_path(frame.image_path, self.normals_dir)
                if aux_path is not None:
                    normals = _load_aux_array(aux_path)
        image, scale_x, scale_y = _resize_image(image, self.downsample_factor)
        if mask is not None:
            mask = _resize_mask(mask, self.downsample_factor)
        if depth is not None:
            depth = _resize_depth(depth, self.downsample_factor)
        if normals is not None:
            normals = _resize_normals(normals, self.downsample_factor)
        intrinsics = frame.render_input.get_intrinsics().clone()
        intrinsics[0, :] *= scale_x
        intrinsics[1, :] *= scale_y

        if self.patch_size is not None:
            image_height, image_width = image.shape[:2]
            random_number_generator = get_numpy_rng()
            crop_start_x = int(
                random_number_generator.integers(
                    0,
                    max(image_width - self.patch_size, 1),
                )
            )
            crop_start_y = int(
                random_number_generator.integers(
                    0,
                    max(image_height - self.patch_size, 1),
                )
            )
            image = image[
                crop_start_y : crop_start_y + self.patch_size,
                crop_start_x : crop_start_x + self.patch_size,
            ]
            if mask is not None:
                mask = mask[
                    crop_start_y : crop_start_y + self.patch_size,
                    crop_start_x : crop_start_x + self.patch_size,
                ]
            if depth is not None:
                depth = depth[
                    crop_start_y : crop_start_y + self.patch_size,
                    crop_start_x : crop_start_x + self.patch_size,
                ]
            if normals is not None:
                normals = normals[
                    crop_start_y : crop_start_y + self.patch_size,
                    crop_start_x : crop_start_x + self.patch_size,
                ]
            intrinsics[0, 2] -= crop_start_x
            intrinsics[1, 2] -= crop_start_y

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = None if mask is None else torch.from_numpy(mask).bool()
        render_input = replace(
            frame.render_input,
            width=image_tensor.shape[1],
            height=image_tensor.shape[0],
            intrinsics=intrinsics,
        )
        sample_kwargs = dict(
            render_input=render_input,
            image=image_tensor,
            image_path=frame.image_path,
            camera_id=frame.camera_id,
            mask=mask_tensor,
            metadata={**frame.metadata, "dataset_index": index},
        )
        if depth is not None:
            depth_tensor = torch.from_numpy(depth).float()
            if depth_tensor.ndim == 2:
                depth_tensor = depth_tensor[..., None]
            if normals is not None:
                normals_tensor = torch.from_numpy(normals[..., :3]).float()
                from gs_utils.data.contract import DepthNormalSample

                return DepthNormalSample(
                    **sample_kwargs,
                    depth=depth_tensor,
                    normals=normals_tensor,
                )
            from gs_utils.data.contract import DepthSample

            return DepthSample(**sample_kwargs, depth=depth_tensor)
        if normals is not None:
            normals_tensor = torch.from_numpy(normals[..., :3]).float()
            from gs_utils.data.contract import NormalSample

            return NormalSample(**sample_kwargs, normals=normals_tensor)
        return DataSample(**sample_kwargs)


def _split_indices(num_images: int, split: str, test_every: int) -> list[int]:
    indices = np.arange(num_images)
    if split == "train":
        return indices[indices % test_every != 0].tolist()
    if split in {"val", "test"}:
        return indices[indices % test_every == 0].tolist()
    if split == "all":
        return indices.tolist()
    raise ValueError(f"Unsupported split: {split}")


def collate_render_inputs(render_inputs: list[RenderInput]) -> RenderInput:
    """Collate render inputs into a single batched render input."""
    reference_render_input = render_inputs[0]
    if any(
        render_input.width != reference_render_input.width
        or render_input.height != reference_render_input.height
        for render_input in render_inputs[1:]
    ):
        raise ValueError(
            "All render inputs in a batch must share width and height."
        )

    if reference_render_input.intrinsics is not None:
        intrinsics = torch.stack(
            [
                render_input.get_intrinsics()
                for render_input in render_inputs
            ],
            dim=0,
        )
        fov = None
    else:
        intrinsics = None
        fov = torch.stack(
            [render_input.get_fov() for render_input in render_inputs],
            dim=0,
        )

    background = None
    if reference_render_input.background is not None:
        background = torch.stack(
            [
                render_input.background
                for render_input in render_inputs
                if render_input.background is not None
            ],
            dim=0,
        )

    return RenderInput(
        cam_to_world=torch.stack(
            [render_input.cam_to_world for render_input in render_inputs],
            dim=0,
        ),
        width=reference_render_input.width,
        height=reference_render_input.height,
        intrinsics=intrinsics,
        fov=fov,
        render_mode=reference_render_input.render_mode,
        background=background,
        metadata={
            "batched_metadata": [
                render_input.metadata for render_input in render_inputs
            ]
        },
    )


def collate_data_samples(batch: list[DataSample]) -> DataSample:
    """Collate typed dataset samples into a batched sample."""
    reference_sample = batch[0]
    sample_kwargs = {
        "render_input": collate_render_inputs(
            [sample.render_input for sample in batch]
        ),
        "image": torch.stack([sample.image for sample in batch], dim=0),
        "image_path": None,
        "camera_id": None,
        "mask": (
            None
            if reference_sample.mask is None
            else torch.stack(
                [
                    sample.mask
                    for sample in batch
                    if sample.mask is not None
                ],
                dim=0,
            )
        ),
        "metadata": {
            "batch_metadata": [sample.metadata for sample in batch],
            "image_paths": [sample.image_path for sample in batch],
            "camera_ids": [sample.camera_id for sample in batch],
        },
    }
    if isinstance(reference_sample, DepthNormalSample):
        return DepthNormalSample(
            **sample_kwargs,
            depth=torch.stack([sample.depth for sample in batch], dim=0),
            normals=torch.stack(
                [sample.normals for sample in batch],
                dim=0,
            ),
        )
    if isinstance(reference_sample, DepthSample):
        return DepthSample(
            **sample_kwargs,
            depth=torch.stack([sample.depth for sample in batch], dim=0),
        )
    if isinstance(reference_sample, NormalSample):
        return NormalSample(
            **sample_kwargs,
            normals=torch.stack(
                [sample.normals for sample in batch],
                dim=0,
            ),
        )
    return DataSample(**sample_kwargs)


def build_dataloader(
    dataset: ParsedSceneDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    """Build a dataloader for typed scene samples."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=collate_data_samples,
    )


def build_dataset(
    parsed_scene: ParsedScene,
    split: str = "train",
    test_every: int = 8,
    patch_size: int | None = None,
    downsample_factor: int = 1,
    preload: bool = False,
    depth_dir: Path | None = None,
    normals_dir: Path | None = None,
) -> ParsedSceneDataset:
    """Build a typed dataset from a parsed scene."""
    indices = _split_indices(
        num_images=len(parsed_scene.frames),
        split=split,
        test_every=test_every,
    )
    return ParsedSceneDataset(
        parsed_scene=parsed_scene,
        indices=indices,
        patch_size=patch_size,
        downsample_factor=downsample_factor,
        preload=preload,
        depth_dir=depth_dir,
        normals_dir=normals_dir,
    )


def _load_parsed_scene(source_config: DataSourceConfig) -> ParsedScene:
    """Load the internal parsed-scene assembly object for a data source."""
    if source_config.type != "colmap":
        raise ValueError(f"Unsupported data source type: {source_config.type}")
    parser = ColmapParser(
        data_dir=str(source_config.path),
        factor=source_config.factor,
        normalize=source_config.normalize,
        test_every=source_config.test_every,
        load_exposure=source_config.load_exposure,
    )
    return parser.to_parsed_scene()


def get_dataset(
    source_config: DataSourceConfig,
    dataset_config: DatasetConfig,
) -> tuple[ParsedSceneDataset, PointCloud | None]:
    """Get a typed dataset and optional point cloud from typed configs."""
    parsed_scene = _load_parsed_scene(source_config)
    dataset = build_dataset(
        parsed_scene=parsed_scene,
        split=dataset_config.split,
        test_every=source_config.test_every,
        patch_size=dataset_config.patch_size,
        downsample_factor=dataset_config.downsample_factor,
        preload=dataset_config.preload,
        depth_dir=dataset_config.depth_dir,
        normals_dir=dataset_config.normals_dir,
    )
    return dataset, parsed_scene.point_cloud
