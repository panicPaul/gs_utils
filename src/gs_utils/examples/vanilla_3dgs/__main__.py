"""CLI entrypoint for the vanilla 3DGS example."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from gs_utils.export import export_ply

from .scene import Vanilla3DGS
from .train import TrainCommand
from .viewer import VanillaViewer


@dataclass
class VisualizeCommand:
    """Load a checkpoint and launch the viewer."""

    checkpoint: Path
    host: str = "0.0.0.0"
    port: int = 8080

    def __call__(self) -> None:
        """Launch the viewer for a saved checkpoint."""
        scene = Vanilla3DGS.load(self.checkpoint)
        VanillaViewer(
            scene=scene,
            output_dir=self.checkpoint.parent,
            host=self.host,
            port=self.port,
        ).launch()


@dataclass
class ExportPlyCommand:
    """Export a checkpoint scene to gsplat-compatible PLY."""

    checkpoint: Path
    output: Path | None = None

    def __call__(self) -> None:
        """Export the checkpoint scene to a PLY file."""
        scene = Vanilla3DGS.load(self.checkpoint)
        output_path = (
            self.output
            if self.output is not None
            else self.checkpoint.with_suffix(".ply")
        )
        export_ply(scene, output_path)
        print(f"Exported PLY to {output_path}")


Vanilla3DGSTrainSubcommand = Annotated[
    TrainCommand,
    tyro.conf.subcommand(name="train"),
]
Vanilla3DGSVisualizeSubcommand = Annotated[
    VisualizeCommand,
    tyro.conf.subcommand(name="visualize"),
]
Vanilla3DGSExportPlySubcommand = Annotated[
    ExportPlyCommand,
    tyro.conf.subcommand(name="export-ply"),
]


def main(
    command: (
        Vanilla3DGSTrainSubcommand
        | Vanilla3DGSVisualizeSubcommand
        | Vanilla3DGSExportPlySubcommand
    ),
) -> None:
    """Vanilla 3DGS with standard densification."""
    command()


if __name__ == "__main__":
    tyro.cli(
        main,
        config=(
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
