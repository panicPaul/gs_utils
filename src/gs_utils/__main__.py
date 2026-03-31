"""Top-level CLI entrypoint for gs_utils."""

import tyro

from .examples.vanilla_3dgs.__main__ import main as vanilla_3dgs_main


def main() -> None:
    """Run the top-level gs_utils CLI."""
    tyro.extras.subcommand_cli_from_dict(
        {
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
