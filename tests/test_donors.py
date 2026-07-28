from open_latent_interfaces.donors import choose_donors
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
