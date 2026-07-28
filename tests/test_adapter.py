import torch

from open_latent_interfaces.adapter import (
    OnlineTransportEnsemble,
    TransportMLP,
    fit_transport_adapter,
    prepare_adapter_projection,
)
from open_latent_interfaces.interventions import OnlineAdapterHook


def test_transport_adapter_fits_signal_and_identity_examples() -> None:
    values = torch.linspace(-2.0, 2.0, 80)
    states = torch.stack((values, torch.sin(values), torch.ones_like(values)), dim=1)
    digits = torch.tensor([index % 10 for index in range(80)])
    identity = torch.tensor([index % 5 == 0 for index in range(80)])
    deltas = torch.stack(
        (
            values * (digits.float() + 1) / 10,
            torch.cos(values) * (digits.float() - 4) / 10,
            torch.zeros_like(values),
        ),
        dim=1,
    )
    deltas[identity] = 0
    projection = prepare_adapter_projection(
        states,
        deltas,
        state_rank=2,
        max_transport_rank=2,
    )
    adapter, history = fit_transport_adapter(
        projection,
        states,
        deltas,
        digits,
        identity,
        hidden_width=32,
        transport_rank=2,
        epochs=120,
        learning_rate=5e-3,
        batch_size=40,
        identity_weight=4.0,
        norm_cap=2.0,
        norm_penalty=0.1,
        seed=7,
        device="cpu",
    )
    predicted = adapter.predict(states, digits)
    baseline_error = float((deltas**2).mean())
    fitted_error = float(((predicted - deltas) ** 2).mean())
    assert history[-1] < history[0]
    assert fitted_error < baseline_error * 0.45
    assert float(predicted[identity].norm(dim=1).mean()) < 0.25


def test_online_adapter_hook_is_differentiable_and_norm_capped() -> None:
    projection = prepare_adapter_projection(
        torch.randn(20, 4),
        torch.randn(20, 4),
        state_rank=2,
        max_transport_rank=2,
    )
    member = TransportMLP(12, 8, 2)
    adapter = OnlineTransportEnsemble(projection, [member], 2)
    hidden = torch.randn(2, 3, 4)
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    hook = OnlineAdapterHook(
        adapter,
        torch.tensor([3, 7]),
        attention_mask,
        scale=4.0,
        norm_cap=0.5,
    )
    modified = hook(None, (), hidden)
    assert hook.applied_delta is not None
    assert hook.recipient_states is not None
    ratios = hook.applied_delta.norm(dim=1) / hook.recipient_states.norm(dim=1)
    assert bool((ratios <= 0.50001).all())
    modified.sum().backward()
    assert all(parameter.grad is not None for parameter in adapter.parameters())


def test_online_adapter_can_train_output_basis() -> None:
    projection = prepare_adapter_projection(
        torch.randn(20, 4),
        torch.randn(20, 4),
        state_rank=2,
        max_transport_rank=2,
    )
    adapter = OnlineTransportEnsemble(projection, [TransportMLP(12, 8, 2)], 2)
    adapter.make_basis_trainable()
    output = adapter(torch.randn(3, 4), torch.tensor([1, 2, 3]))
    output.square().mean().backward()
    assert isinstance(adapter.delta_basis, torch.nn.Parameter)
    assert adapter.delta_basis.grad is not None
