"""Initialization strategy registry."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from gs_utils.initialization.common import InitContext, InitScene
from gs_utils.initialization.config import InitializationConfig

SceneT = TypeVar("SceneT", bound=InitScene)
TypedInitFn = Callable[[SceneT, InitializationConfig, InitContext], None]


@dataclass(slots=True, frozen=True)
class InitRegistration[SceneT: InitScene]:
    """Registered initialization strategy metadata."""

    scene_type: type[SceneT]
    init_fn: TypedInitFn[SceneT]


INIT_FNS: dict[str, list[InitRegistration[InitScene]]] = {}


def register_init_fn(
    *names: str, scene_type: type[SceneT]
) -> Callable[[TypedInitFn[SceneT]], TypedInitFn[SceneT]]:
    """Register an initialization strategy under one or more names."""

    def decorator(init_fn: TypedInitFn[SceneT]) -> TypedInitFn[SceneT]:
        for strategy_name in names:
            strategy_registrations = INIT_FNS.setdefault(strategy_name, [])
            if any(
                registration.scene_type == scene_type
                for registration in strategy_registrations
            ):
                raise ValueError(
                    f"Duplicate initialization strategy for {strategy_name!r} and "
                    f"{scene_type!r}."
                )
            strategy_registrations.append(
                InitRegistration(scene_type=scene_type, init_fn=init_fn)
            )
        return init_fn

    return decorator
