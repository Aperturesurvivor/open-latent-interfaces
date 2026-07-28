from __future__ import annotations

from dataclasses import dataclass

import torch
from safetensors.torch import load_file
from torch import nn


class TransportMLP(nn.Module):
    """Small bottlenecked map from native state coordinates and a digit."""

    def __init__(self, input_width: int, hidden_width: int, output_width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, output_width),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class OnlineTransportEnsemble(nn.Module):
    """Differentiable ensemble injected into a frozen model forward pass."""

    def __init__(
        self,
        projection: AdapterProjection,
        members: list[TransportMLP],
        transport_rank: int,
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.transport_rank = transport_rank
        self.register_buffer("state_mean", projection.state_mean)
        self.register_buffer("state_basis", projection.state_basis)
        self.register_buffer("state_scale", projection.state_scale)
        self.register_buffer(
            "delta_basis",
            projection.delta_basis[:transport_rank],
        )
        self.register_buffer(
            "coefficient_scale",
            projection.coefficient_scale[:transport_rank],
        )

    def forward(
        self,
        states: torch.Tensor,
        target_digits: torch.Tensor,
    ) -> torch.Tensor:
        states = states.float()
        scores = ((states - self.state_mean) @ self.state_basis.T) / self.state_scale
        one_hot = torch.nn.functional.one_hot(
            target_digits,
            num_classes=10,
        ).float()
        features = torch.cat((scores, one_hot), dim=1)
        standardized = torch.stack([member(features) for member in self.members]).mean(
            dim=0
        )
        coefficients = standardized * self.coefficient_scale
        return coefficients @ self.delta_basis

    def make_basis_trainable(self) -> None:
        if isinstance(self.delta_basis, nn.Parameter):
            return
        basis = self._buffers.pop("delta_basis")
        self.register_parameter("delta_basis", nn.Parameter(basis))

    @torch.inference_mode()
    def predict(
        self,
        states: torch.Tensor,
        target_digits: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        return self(states.to(device), target_digits.to(device)).float().cpu()


def load_online_adapter(
    path: str,
    *,
    step: int,
    member_count: int = 3,
) -> OnlineTransportEnsemble:
    tensors = load_file(path)
    prefix = f"step{step}"
    projection = AdapterProjection(
        state_mean=tensors[f"{prefix}.projection.state_mean"],
        state_basis=tensors[f"{prefix}.projection.state_basis"],
        state_scale=tensors[f"{prefix}.projection.state_scale"],
        delta_basis=tensors[f"{prefix}.projection.delta_basis"],
        coefficient_scale=tensors[f"{prefix}.projection.coefficient_scale"],
    )
    members = []
    transport_rank = 0
    for member_index in range(member_count):
        member_prefix = f"{prefix}.member{member_index}.model"
        first_weight = tensors[f"{member_prefix}.network.0.weight"]
        second_weight = tensors[f"{member_prefix}.network.2.weight"]
        hidden_width = first_weight.shape[0]
        transport_rank = second_weight.shape[0]
        member = TransportMLP(first_weight.shape[1], hidden_width, transport_rank)
        member.load_state_dict(
            {
                "network.0.weight": first_weight,
                "network.0.bias": tensors[f"{member_prefix}.network.0.bias"],
                "network.2.weight": second_weight,
                "network.2.bias": tensors[f"{member_prefix}.network.2.bias"],
            }
        )
        members.append(member)
    return OnlineTransportEnsemble(projection, members, transport_rank)


@dataclass(frozen=True)
class AdapterProjection:
    state_mean: torch.Tensor
    state_basis: torch.Tensor
    state_scale: torch.Tensor
    delta_basis: torch.Tensor
    coefficient_scale: torch.Tensor

    def features(self, states: torch.Tensor, target_digits: torch.Tensor) -> torch.Tensor:
        if states.ndim != 2 or target_digits.shape != (states.shape[0],):
            raise ValueError("states and target digits must align")
        scores = ((states - self.state_mean) @ self.state_basis.T) / self.state_scale
        one_hot = torch.nn.functional.one_hot(target_digits, num_classes=10).float()
        return torch.cat((scores, one_hot), dim=1)

    def coefficient_targets(
        self,
        deltas: torch.Tensor,
        *,
        transport_rank: int,
    ) -> torch.Tensor:
        return (deltas @ self.delta_basis[:transport_rank].T) / self.coefficient_scale[
            :transport_rank
        ]


@dataclass
class FittedTransportAdapter:
    projection: AdapterProjection
    model: TransportMLP
    transport_rank: int

    @torch.inference_mode()
    def predict(
        self,
        states: torch.Tensor,
        target_digits: torch.Tensor,
    ) -> torch.Tensor:
        self.model.eval()
        features = self.projection.features(states, target_digits)
        standardized = self.model(features)
        coefficients = (
            standardized * self.projection.coefficient_scale[: self.transport_rank]
        )
        return coefficients @ self.projection.delta_basis[: self.transport_rank]


def prepare_adapter_projection(
    states: torch.Tensor,
    deltas: torch.Tensor,
    *,
    state_rank: int,
    max_transport_rank: int,
) -> AdapterProjection:
    if states.ndim != 2 or deltas.shape != states.shape:
        raise ValueError("states and deltas must be aligned matrices")
    if state_rank < 1 or state_rank > min(states.shape):
        raise ValueError("state rank is outside the state matrix")
    if max_transport_rank < 1 or max_transport_rank > min(deltas.shape):
        raise ValueError("transport rank is outside the delta matrix")
    state_mean = states.mean(dim=0)
    _, _, state_basis = torch.linalg.svd(states - state_mean, full_matrices=False)
    state_basis = state_basis[:state_rank]
    state_scores = (states - state_mean) @ state_basis.T
    state_scale = state_scores.std(dim=0).clamp_min(1e-6)
    _, _, delta_basis = torch.linalg.svd(deltas, full_matrices=False)
    delta_basis = delta_basis[:max_transport_rank]
    coefficients = deltas @ delta_basis.T
    coefficient_scale = coefficients.std(dim=0).clamp_min(1e-6)
    return AdapterProjection(
        state_mean=state_mean,
        state_basis=state_basis,
        state_scale=state_scale,
        delta_basis=delta_basis,
        coefficient_scale=coefficient_scale,
    )


def fit_transport_adapter(
    projection: AdapterProjection,
    states: torch.Tensor,
    deltas: torch.Tensor,
    target_digits: torch.Tensor,
    identity_mask: torch.Tensor,
    *,
    hidden_width: int,
    transport_rank: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    identity_weight: float,
    norm_cap: float,
    norm_penalty: float,
    seed: int,
    device: torch.device | str,
) -> tuple[FittedTransportAdapter, list[float]]:
    if identity_mask.shape != (states.shape[0],):
        raise ValueError("identity mask must align with states")
    if min(epochs, batch_size) < 1:
        raise ValueError("epochs and batch size must be positive")
    torch.manual_seed(seed)
    device = torch.device(device)
    features = projection.features(states, target_digits)
    targets = projection.coefficient_targets(
        deltas,
        transport_rank=transport_rank,
    )
    input_width = features.shape[1]
    model = TransportMLP(input_width, hidden_width, transport_rank).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    basis = projection.delta_basis[:transport_rank].to(device)
    coefficient_scale = projection.coefficient_scale[:transport_rank].to(device)
    state_norms = states.norm(dim=1).clamp_min(1e-6)
    weights = torch.where(
        identity_mask,
        torch.tensor(identity_weight),
        torch.tensor(1.0),
    )
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        permutation = torch.randperm(states.shape[0], generator=generator)
        epoch_loss = 0.0
        examples_seen = 0
        model.train()
        for start in range(0, states.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            batch_features = features[indices].to(device)
            batch_targets = targets[indices].to(device)
            batch_weights = weights[indices].to(device)
            predicted = model(batch_features)
            coefficient_error = ((predicted - batch_targets) ** 2).mean(dim=1)
            raw_coefficients = predicted * coefficient_scale
            predicted_delta = raw_coefficients @ basis
            relative_norm = predicted_delta.norm(dim=1) / state_norms[indices].to(
                device
            )
            excess_norm = torch.relu(relative_norm - norm_cap) ** 2
            loss = (coefficient_error * batch_weights).mean()
            loss = loss + norm_penalty * excess_norm.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * indices.numel()
            examples_seen += indices.numel()
        history.append(epoch_loss / examples_seen)
    model = model.cpu()
    return (
        FittedTransportAdapter(
            projection=projection,
            model=model,
            transport_rank=transport_rank,
        ),
        history,
    )
