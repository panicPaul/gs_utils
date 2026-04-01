"""Top-level CLI entrypoint for gs_utils."""

import tyro

from .evaluation import EvaluateCommand
from .examples.vanilla_3dgs.__main__ import main as vanilla_3dgs_main
from .viewer import ViewPlyCommand


def main() -> None:
    """Run the top-level gs_utils CLI."""
    tyro.extras.subcommand_cli_from_dict(
        {
            "evaluate": EvaluateCommand,
            "view-ply": ViewPlyCommand,
            "vanilla_3dgs": vanilla_3dgs_main,
        },
        config=(
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
        use_underscores=True,
    )


if __name__ == "__main__":
    main()
