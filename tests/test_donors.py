from open_latent_interfaces.donors import choose_donors, choose_multi_donors
from open_latent_interfaces.phase1_data import build_phase1_additions


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
