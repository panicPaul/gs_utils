"""COLMAP adapters for shared parsed-scene contracts."""

from pathlib import Path

import torch

from gs_utils.contracts import RenderInput
from gs_utils.data.colmap.colmap import Parser
from gs_utils.data.contract import ParsedScene, PointCloud, SceneFrame


class ColmapParser(Parser):
    """COLMAP parser exposing shared parsed-scene conversion helpers."""

    def to_parsed_scene(self) -> ParsedScene:
        point_cloud = None
        if len(self.points) > 0:
            colors = torch.from_numpy(self.points_rgb).float() / 255.0
            point_cloud = PointCloud(
                positions=torch.from_numpy(self.points).float(),
                colors=colors,
            )
        frames = [
            SceneFrame(
                render_input=RenderInput(
                    cam_to_world=torch.from_numpy(
                        self.camtoworlds[index]
                    ).float(),
                    width=self.imsize_dict[camera_id][0],
                    height=self.imsize_dict[camera_id][1],
                    intrinsics=torch.from_numpy(
                        self.Ks_dict[camera_id]
                    ).float(),
                ),
                image_path=Path(self.image_paths[index]),
                camera_id=camera_id,
                mask=(
                    None
                    if self.mask_dict[camera_id] is None
                    else torch.from_numpy(self.mask_dict[camera_id]).bool()
                ),
                metadata={
                    "image_name": self.image_names[index],
                    "dataset_index": index,
                },
            )
            for index, camera_id in enumerate(self.camera_ids)
        ]
        return ParsedScene(
            frames=frames,
            scene_scale=float(self.scene_scale),
            normalization_transform=torch.from_numpy(self.transform).float(),
            point_cloud=point_cloud,
        )
