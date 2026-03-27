"""Initialization strategy registry."""

from collections.abc import Callable
from dataclasses import dataclass

from gs_utils.initialization.common import InitFn


@dataclass(slots=True, frozen=True)
class InitRegistration:
    """Registered initialization strategy metadata."""

    scene_type: type[object]
    init_fn: InitFn


INIT_FNS: dict[str, list[InitRegistration]] = {}


def register_init_fn(
    *names: str, scene_type: type[object]
) -> Callable[[InitFn], InitFn]:
    """Register an initialization strategy under one or more names."""

    def decorator(init_fn: InitFn) -> InitFn:
        for name in names:
            registrations = INIT_FNS.setdefault(name, [])
            if any(reg.scene_type == scene_type for reg in registrations):
                raise ValueError(
                    f"Duplicate initialization strategy for {name!r} and "
                    f"{scene_type!r}."
                )
            registrations.append(
                InitRegistration(scene_type=scene_type, init_fn=init_fn)
            )
        return init_fn

    return decorator
