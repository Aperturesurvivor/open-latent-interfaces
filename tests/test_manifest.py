from open_latent_interfaces.manifest import stable_json_sha256


def test_manifest_hash_is_key_order_independent() -> None:
    assert stable_json_sha256({"a": 1, "b": 2}) == stable_json_sha256({"b": 2, "a": 1})

