"""
Tests for deltacert.signing — the mechanism that turns a cert JSON from an
editable file into a tamper-evident artifact.

These are adversarial by design: a signing scheme is only as good as the
attacks it rejects, so most of this file is "does verification correctly
FAIL" rather than "does it pass". A signing test suite that only exercises
the happy path is not a signing test suite.
"""

import base64
import copy
import glob
import json
import os

import pytest

from deltacert import signing as dsign
from deltacert.collectors import validation_status_for_layers

try:
    import jsonschema
    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
SCHEMA_PATH = os.path.join(REPO_ROOT, "deltacert-schema.json")
VALIDATION_RESULTS_DIR = os.path.join(REPO_ROOT, "validation_results")


@pytest.fixture
def keypair(tmp_path):
    priv_pem, pub_pem = dsign.generate_keypair()
    priv_path = tmp_path / "priv.pem"
    pub_path = tmp_path / "pub.pem"
    priv_path.write_bytes(priv_pem)
    pub_path.write_bytes(pub_pem)
    return dsign.load_private_key(str(priv_path)), dsign.load_public_key(str(pub_path))


@pytest.fixture
def sample_cert():
    return {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "certified": True,
        "threshold_d": 0.5,
        "formula": "d_COMM = -log(E[4c*sqrt(1-c^2)] / 2), certified if d >= threshold",
        "validation_status": "flagship_validated",
        "layers": {
            "weight_quant": {"d_comm": 1.153, "certified": True, "budget": 0.5},
        },
    }


# ─────────────────────────────────────────────────────────────────────────
# Round trip
# ─────────────────────────────────────────────────────────────────────────

def test_sign_then_verify_succeeds(keypair, sample_cert):
    private_key, public_key = keypair
    signed = dsign.sign_certificate(sample_cert, private_key, key_id="test")
    result = dsign.verify_certificate(signed, public_key)
    assert result.ok
    assert bool(result) is True


def test_signing_does_not_mutate_input(keypair, sample_cert):
    private_key, _ = keypair
    original = copy.deepcopy(sample_cert)
    dsign.sign_certificate(sample_cert, private_key)
    assert sample_cert == original, "sign_certificate must not mutate its input"


def test_signed_cert_has_expected_signature_shape(keypair, sample_cert):
    private_key, _ = keypair
    signed = dsign.sign_certificate(sample_cert, private_key, key_id="my-key")
    sig = signed["signature"]
    assert sig["alg"] == "Ed25519"
    assert sig["key_id"] == "my-key"
    assert isinstance(sig["value"], str)
    # value must be valid base64
    base64.b64decode(sig["value"], validate=True)


def test_resigning_replaces_rather_than_nests(keypair, sample_cert):
    private_key, public_key = keypair
    once = dsign.sign_certificate(sample_cert, private_key, key_id="v1")
    twice = dsign.sign_certificate(once, private_key, key_id="v2")
    assert twice["signature"]["key_id"] == "v2"
    result = dsign.verify_certificate(twice, public_key)
    assert result.ok


# ─────────────────────────────────────────────────────────────────────────
# Tamper detection — the actual point of this feature
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("field,new_value", [
    ("certified", False),
    ("threshold_d", 99.0),
    ("model", "a-different-model"),
    ("validation_status", "flagship_validated"),  # even flipping TO the "better" status must fail
])
def test_tampering_top_level_field_breaks_verification(keypair, sample_cert, field, new_value):
    private_key, public_key = keypair
    sample_cert["validation_status"] = "implemented_pending_validation"
    signed = dsign.sign_certificate(sample_cert, private_key)
    tampered = copy.deepcopy(signed)
    tampered[field] = new_value
    result = dsign.verify_certificate(tampered, public_key)
    assert not result.ok
    assert "modified" in result.reason or "wrong key" in result.reason


def test_tampering_nested_d_comm_breaks_verification(keypair, sample_cert):
    private_key, public_key = keypair
    signed = dsign.sign_certificate(sample_cert, private_key)
    tampered = copy.deepcopy(signed)
    tampered["layers"]["weight_quant"]["d_comm"] = 999.0
    result = dsign.verify_certificate(tampered, public_key)
    assert not result.ok


def test_stripping_validation_status_breaks_verification(keypair, sample_cert):
    """The whole point of signing validation_status: you cannot delete the
    'this collector isn't flagship-validated yet' label and still pass."""
    private_key, public_key = keypair
    sample_cert["validation_status"] = "implemented_pending_validation"
    signed = dsign.sign_certificate(sample_cert, private_key)
    tampered = copy.deepcopy(signed)
    del tampered["validation_status"]
    result = dsign.verify_certificate(tampered, public_key)
    assert not result.ok


def test_adding_extra_field_breaks_verification(keypair, sample_cert):
    private_key, public_key = keypair
    signed = dsign.sign_certificate(sample_cert, private_key)
    tampered = copy.deepcopy(signed)
    tampered["extra_injected_field"] = "attacker-controlled"
    result = dsign.verify_certificate(tampered, public_key)
    assert not result.ok


def test_reordering_keys_does_not_break_verification(keypair, sample_cert):
    """Canonicalization must be order-independent — this is NOT tampering."""
    private_key, public_key = keypair
    signed = dsign.sign_certificate(sample_cert, private_key)
    reordered = json.loads(json.dumps(dict(reversed(list(signed.items())))))
    result = dsign.verify_certificate(reordered, public_key)
    assert result.ok


# ─────────────────────────────────────────────────────────────────────────
# Wrong key / unsigned / malformed inputs
# ─────────────────────────────────────────────────────────────────────────

def test_wrong_public_key_fails(keypair, sample_cert, tmp_path):
    private_key, _ = keypair
    other_priv_pem, other_pub_pem = dsign.generate_keypair()
    other_pub_path = tmp_path / "other_pub.pem"
    other_pub_path.write_bytes(other_pub_pem)
    other_public_key = dsign.load_public_key(str(other_pub_path))

    signed = dsign.sign_certificate(sample_cert, private_key)
    result = dsign.verify_certificate(signed, other_public_key)
    assert not result.ok


def test_unsigned_cert_fails_with_clear_reason(keypair, sample_cert):
    _, public_key = keypair
    result = dsign.verify_certificate(sample_cert, public_key)
    assert not result.ok
    assert "never signed" in result.reason


def test_malformed_signature_block_not_a_dict(keypair, sample_cert):
    _, public_key = keypair
    cert = dict(sample_cert)
    cert["signature"] = "not-a-dict"
    result = dsign.verify_certificate(cert, public_key)
    assert not result.ok


def test_signature_with_wrong_algorithm_rejected(keypair, sample_cert):
    _, public_key = keypair
    cert = dict(sample_cert)
    cert["signature"] = {"alg": "HMAC-SHA256", "value": "deadbeef"}
    result = dsign.verify_certificate(cert, public_key)
    assert not result.ok
    assert "unsupported" in result.reason.lower()


def test_signature_value_not_valid_base64(keypair, sample_cert):
    _, public_key = keypair
    cert = dict(sample_cert)
    cert["signature"] = {"alg": "Ed25519", "value": "!!!not-base64!!!"}
    result = dsign.verify_certificate(cert, public_key)
    assert not result.ok
    assert "base64" in result.reason.lower()


def test_signature_value_missing(keypair, sample_cert):
    _, public_key = keypair
    cert = dict(sample_cert)
    cert["signature"] = {"alg": "Ed25519"}
    result = dsign.verify_certificate(cert, public_key)
    assert not result.ok


def test_verify_non_dict_certificate(keypair):
    _, public_key = keypair
    result = dsign.verify_certificate(["not", "a", "cert"], public_key)
    assert not result.ok


def test_canonicalize_rejects_non_dict():
    with pytest.raises(dsign.SigningError):
        dsign.canonicalize("not a dict")


# ─────────────────────────────────────────────────────────────────────────
# Canonicalization determinism
# ─────────────────────────────────────────────────────────────────────────

def test_canonicalize_excludes_signature_field():
    cert = {"a": 1, "signature": {"alg": "Ed25519", "value": "xyz"}}
    canon = dsign.canonicalize(cert)
    assert b"signature" not in canon


def test_canonicalize_is_key_order_independent():
    cert_a = {"b": 2, "a": 1}
    cert_b = {"a": 1, "b": 2}
    assert dsign.canonicalize(cert_a) == dsign.canonicalize(cert_b)


def test_canonicalize_is_whitespace_independent_of_original_formatting(tmp_path):
    cert = {"a": 1, "b": {"c": 2}}
    compact = json.dumps(cert, separators=(",", ":"))
    spaced = json.dumps(cert, indent=4)
    assert dsign.canonicalize(json.loads(compact)) == dsign.canonicalize(json.loads(spaced))


# ─────────────────────────────────────────────────────────────────────────
# Key loading
# ─────────────────────────────────────────────────────────────────────────

def test_load_private_key_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        dsign.load_private_key(str(tmp_path / "nonexistent.pem"))


def test_load_private_key_rejects_garbage_file(tmp_path):
    bad = tmp_path / "bad.pem"
    bad.write_text("this is not a PEM key")
    with pytest.raises(dsign.SigningError):
        dsign.load_private_key(str(bad))


def test_load_public_key_rejects_private_key_file(tmp_path):
    priv_pem, _ = dsign.generate_keypair()
    priv_path = tmp_path / "priv.pem"
    priv_path.write_bytes(priv_pem)
    with pytest.raises(dsign.SigningError):
        dsign.load_public_key(str(priv_path))


def test_load_private_key_rejects_public_key_file(tmp_path):
    _, pub_pem = dsign.generate_keypair()
    pub_path = tmp_path / "pub.pem"
    pub_path.write_bytes(pub_pem)
    with pytest.raises(dsign.SigningError):
        dsign.load_private_key(str(pub_path))


# ─────────────────────────────────────────────────────────────────────────
# validation_status stamping (collectors.py)
# ─────────────────────────────────────────────────────────────────────────

def test_validation_status_all_flagship_collectors():
    from deltacert.collectors import validation_status_for_layers
    assert validation_status_for_layers(["weight_quant"]) == "flagship_validated"
    assert validation_status_for_layers(["weight_quant", "engine_swap"]) == "flagship_validated"


def test_validation_status_mixed_collectors_fails_toward_unvalidated():
    from deltacert.collectors import validation_status_for_layers
    assert validation_status_for_layers(["weight_quant", "lora"]) == "implemented_pending_validation"


def test_validation_status_all_unvalidated():
    from deltacert.collectors import validation_status_for_layers
    assert validation_status_for_layers(["lora", "prefix_cache"]) == "implemented_pending_validation"


def test_validation_status_empty_layers():
    from deltacert.collectors import validation_status_for_layers
    assert validation_status_for_layers([]) == "implemented_pending_validation"


def test_flagship_list_matches_paper_exactly():
    """The 7 flagships this constant names must match the paper's Table 1
    exactly — this test is a tripwire against silent drift if someone edits
    the set without checking it against the published claims."""
    from deltacert.collectors import FLAGSHIP_VALIDATED_COLLECTORS
    expected = {
        "weight_quant", "kv_cache_quant", "engine_swap", "batch_divergence",
        "spec_decoding", "provider_drift", "trajectory",
    }
    assert FLAGSHIP_VALIDATED_COLLECTORS == expected
    assert len(FLAGSHIP_VALIDATED_COLLECTORS) == 7


# ─────────────────────────────────────────────────────────────────────────
# Real on-disk certs: schema validation + live signature check
# ─────────────────────────────────────────────────────────────────────────

def _real_cert_files():
    return sorted(glob.glob(os.path.join(VALIDATION_RESULTS_DIR, "**", "cert_*.json"), recursive=True))


@pytest.mark.skipif(not HAVE_JSONSCHEMA, reason="jsonschema not installed")
@pytest.mark.parametrize("cert_path", _real_cert_files())
def test_real_cert_validates_against_published_schema(cert_path):
    schema = json.load(open(SCHEMA_PATH))
    cert = json.load(open(cert_path))
    jsonschema.validate(cert, schema)  # raises on failure


@pytest.mark.parametrize("cert_path", _real_cert_files())
def test_real_cert_has_flagship_validation_status(cert_path):
    """Every real cert's validation_status must match what
    validation_status_for_layers computes for its layers — the same function
    that produces the field in the first place. This catches both a cert
    left with a stale/hand-edited status and a genuinely non-flagship
    collector (e.g. trajectory_kv_fp8, a §5.5-only check) being mislabeled
    as flagship_validated."""
    cert = json.load(open(cert_path))
    expected = validation_status_for_layers(cert.get("layers", {}).keys())
    assert cert.get("validation_status") == expected


def test_real_certs_verify_against_repo_public_key():
    """End-to-end trust-chain check: every published cert must actually
    verify against the public key committed to the repo, not just contain
    a well-formed-looking signature block."""
    pub_key_path = os.path.join(REPO_ROOT, "deltacert-public.pem")
    if not os.path.exists(pub_key_path):
        pytest.skip("deltacert-public.pem not present (run scripts/backfill_signatures.py --keygen)")
    public_key = dsign.load_public_key(pub_key_path)
    failures = []
    for f in _real_cert_files():
        cert = json.load(open(f))
        result = dsign.verify_certificate(cert, public_key)
        if not result.ok:
            failures.append((f, result.reason))
    assert not failures, f"Certs that fail verification against the repo public key: {failures}"


def test_real_certs_are_not_accidentally_left_unsigned():
    """This is a repo-state check, not a pure unit test: if someone adds a
    new cert file without running the backfill/sign step, this should be
    the thing that catches it before it ships."""
    unsigned = [f for f in _real_cert_files() if "signature" not in json.load(open(f))]
    assert not unsigned, f"Unsigned cert files found: {unsigned}"
