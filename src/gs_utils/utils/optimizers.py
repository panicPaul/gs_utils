"""Low-level optimizer and parameter update helpers."""

from collections.abc import Callable

import torch
from torch import nn
from torch.optim import Optimizer


def build_append_selected_parameter_factory(
    selected_indices: torch.Tensor,
) -> Callable[[str, torch.Tensor], nn.Parameter]:
    """Return a factory that appends selected rows to each parameter tensor."""

    def append_selected_parameter_rows(
        parameter_name: str,
        current_parameter: torch.Tensor,
    ) -> nn.Parameter:
        """Append selected parameter rows to the end of the tensor."""
        del parameter_name
        return nn.Parameter(
            torch.cat([current_parameter, current_parameter[selected_indices]]),
            requires_grad=current_parameter.requires_grad,
        )

    return append_selected_parameter_rows


def build_append_zero_optimizer_state_factory(
    selected_indices: torch.Tensor,
) -> Callable[[str, torch.Tensor], torch.Tensor]:
    """Return a factory that appends zero-initialized optimizer state rows."""

    def append_zero_optimizer_state_rows(
        optimizer_state_name: str,
        optimizer_state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Append zero-initialized state rows for newly created parameters."""
        del optimizer_state_name
        return torch.cat(
            [
                optimizer_state_tensor,
                torch.zeros(
                    (
                        selected_indices.numel(),
                        *optimizer_state_tensor.shape[1:],
                    ),
                    device=optimizer_state_tensor.device,
                ),
            ]
        )

    return append_zero_optimizer_state_rows


def build_keep_selected_parameter_factory(
    kept_indices: torch.Tensor,
) -> Callable[[str, torch.Tensor], nn.Parameter]:
    """Return a factory that keeps only the selected rows of each parameter."""

    def keep_selected_parameter_rows(
        parameter_name: str,
        current_parameter: torch.Tensor,
    ) -> nn.Parameter:
        """Filter a parameter tensor down to the kept rows."""
        del parameter_name
        return nn.Parameter(
            current_parameter[kept_indices],
            requires_grad=current_parameter.requires_grad,
        )

    return keep_selected_parameter_rows


def build_keep_selected_optimizer_state_factory(
    kept_indices: torch.Tensor,
) -> Callable[[str, torch.Tensor], torch.Tensor]:
    """Return a factory that keeps only the selected optimizer-state rows."""

    def keep_selected_optimizer_state_rows(
        optimizer_state_name: str,
        optimizer_state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Filter an optimizer-state tensor down to the kept rows."""
        del optimizer_state_name
        return optimizer_state_tensor[kept_indices]

    return keep_selected_optimizer_state_rows


def build_zero_optimizer_state_factory() -> Callable[
    [str, torch.Tensor], torch.Tensor
]:
    """Return a factory that zeros an optimizer-state tensor in place shape-wise."""

    def zero_optimizer_state(
        optimizer_state_name: str,
        optimizer_state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Reset optimizer-state tensors to zero while keeping their shape."""
        del optimizer_state_name
        return torch.zeros_like(optimizer_state_tensor)

    return zero_optimizer_state


def remap_parameters_and_optimizer_state(
    *,
    parameters_by_name: dict[str, nn.Parameter],
    optimizers_by_parameter_name: dict[str, Optimizer],
    updated_parameter_factory: Callable[[str, torch.Tensor], nn.Parameter],
    updated_optimizer_state_factory: Callable[
        [str, torch.Tensor], torch.Tensor
    ],
    parameter_names: tuple[str, ...] | None = None,
) -> dict[str, nn.Parameter]:
    """Apply a shared remap to parameters and their optimizer state."""
    updated_parameters_by_name = dict(parameters_by_name)
    names_to_update = (
        parameter_names
        if parameter_names is not None
        else tuple(parameters_by_name.keys())
    )
    for parameter_name in names_to_update:
        current_parameter = parameters_by_name[parameter_name]
        updated_parameter = updated_parameter_factory(
            parameter_name,
            current_parameter,
        )
        updated_parameters_by_name[parameter_name] = updated_parameter
        if parameter_name not in optimizers_by_parameter_name:
            continue

        optimizer = optimizers_by_parameter_name[parameter_name]
        optimizer_state = optimizer.state[current_parameter]
        if current_parameter in optimizer.state:
            del optimizer.state[current_parameter]
        updated_optimizer_state: dict[str, torch.Tensor | int] = {}
        for (
            optimizer_state_name,
            optimizer_state_value,
        ) in optimizer_state.items():
            if optimizer_state_name == "step":
                updated_optimizer_state[optimizer_state_name] = (
                    optimizer_state_value
                )
            else:
                updated_optimizer_state[optimizer_state_name] = (
                    updated_optimizer_state_factory(
                        optimizer_state_name,
                        optimizer_state_value,
                    )
                )
        for parameter_group in optimizer.param_groups:
            parameter_group["params"] = [updated_parameter]
        optimizer.state[updated_parameter] = updated_optimizer_state
    return updated_parameters_by_name
