"""CLI entrypoint for the vanilla 3DGS example."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from .train import TrainCommand


@dataclass
class VisualizeCommand:
    """Load a checkpoint and launch the viewer."""

    checkpoint: Path

    def __call__(self) -> None:
        """Launch the viewer for a saved checkpoint."""
        raise NotImplementedError


Vanilla3DGSTrainSubcommand = Annotated[
    TrainCommand,
    tyro.conf.subcommand(name="train"),
]
Vanilla3DGSVisualizeSubcommand = Annotated[
    VisualizeCommand,
    tyro.conf.subcommand(name="visualize"),
]


def main(
    command: Vanilla3DGSTrainSubcommand | Vanilla3DGSVisualizeSubcommand,
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
