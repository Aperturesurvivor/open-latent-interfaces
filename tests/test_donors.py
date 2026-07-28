from open_latent_interfaces.donors import (
    choose_cyclic_donors,
    choose_donors,
    choose_multi_donors,
    choose_position_donors,
    choose_prefix_donors,
)
from open_latent_interfaces.phase1_data import build_phase1_additions
from open_latent_interfaces.phase2_data import (
    balanced_counterfactual_results,
    build_phase2_additions,
)


def test_donor_selection_changes_or_preserves_leading_digit() -> None:
    examples = [
        example
        for example in build_phase1_additions()
        if example.split == "development"
    ]
    targeted, same = choose_donors(examples)
    for recipient, target_index, same_index in zip(
        examples, targeted, same, strict=True
    ):
        original_digit = int(str(recipient.result)[0])
        assert int(str(examples[target_index].result)[0]) == original_digit % 9 + 1
        assert int(str(examples[same_index].result)[0]) == original_digit
        assert examples[same_index].example_id != recipient.example_id


def test_multi_donors_cover_every_alternative_leading_digit() -> None:
    examples = [
        example
        for example in build_phase1_additions()
        if example.split == "train"
    ][:60]
    selections = choose_multi_donors(examples)
    for recipient, donor_indices in zip(examples, selections, strict=True):
        original_digit = int(str(recipient.result)[0])
        observed = {int(str(examples[index].result)[0]) for index in donor_indices}
        assert len(donor_indices) == 8
        assert observed == set(range(1, 10)) - {original_digit}


def test_cyclic_donors_follow_requested_offsets() -> None:
    examples = [
        example
        for example in build_phase1_additions()
        if example.split == "train"
    ]
    selections = choose_cyclic_donors(examples, offsets=(1, 3, 5, 7))
    for recipient, donor_indices in zip(examples, selections, strict=True):
        original = int(str(recipient.result)[0])
        expected = {(original - 1 + offset) % 9 + 1 for offset in (1, 3, 5, 7)}
        observed = {int(str(examples[index].result)[0]) for index in donor_indices}
        assert observed == expected


def test_prefix_donors_match_targets_and_are_pool_order_invariant() -> None:
    examples = build_phase2_additions()
    fit = [example for example in examples if example.split == "fit"]
    selection = [example for example in examples if example.split == "selection"]
    targets = balanced_counterfactual_results(selection)
    forward = choose_prefix_donors(
        fit,
        selection,
        targets,
        prefix_length=2,
    )
    reversed_fit = list(reversed(fit))
    backward = choose_prefix_donors(
        reversed_fit,
        selection,
        targets,
        prefix_length=2,
    )
    forward_ids = [fit[index].example_id for index in forward]
    backward_ids = [reversed_fit[index].example_id for index in backward]
    assert forward_ids == backward_ids
    assert all(
        str(fit[index].result).startswith(str(target)[:2])
        for index, target in zip(forward, targets, strict=True)
    )


def test_position_donors_match_requested_or_wrong_digit() -> None:
    examples = build_phase2_additions()
    fit = [example for example in examples if example.split == "fit"]
    selection = [example for example in examples if example.split == "selection"]
    targets = balanced_counterfactual_results(selection)
    for position in range(3):
        targeted = choose_position_donors(
            fit,
            selection,
            targets,
            position=position,
        )
        wrong = choose_position_donors(
            fit,
            selection,
            targets,
            position=position,
            wrong_digit=True,
        )
        for target, targeted_index, wrong_index in zip(
            targets,
            targeted,
            wrong,
            strict=True,
        ):
            desired = int(str(target)[position])
            assert int(str(fit[targeted_index].result)[position]) == desired
            assert int(str(fit[wrong_index].result)[position]) != desired
