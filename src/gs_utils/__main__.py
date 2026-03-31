"""Top-level CLI entrypoint for gs_utils."""

import tyro

from .examples.vanilla_3dgs.__main__ import main as vanilla_3dgs_main

if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "vanilla-3dgs": vanilla_3dgs_main,
        },
        config=(
            tyro.conf.CascadeSubcommandArgs,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
