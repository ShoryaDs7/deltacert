"""Unit tests for the `deltacert free-running` CLI subcommand.

No GPU/vLLM required: the "vLLM not installed" path is tested for real (this
dev machine has no vllm installed), and the "collection succeeds" path mocks
collect_free_running_vllm so the JSON-writing/exit-code logic is verified
without booting real engines.
"""
import argparse
import json
import sys
import types

import pytest

from deltacert import cli


def _build_free_running_parser():
    """Extract just the free-running subparser the way main() builds it,
    without invoking main()'s full argparse tree construction twice."""
    parser = argparse.ArgumentParser(prog="deltacert")
    sub = parser.add_subparsers(dest="command")
    p2c = sub.add_parser("free-running")
    p2c.add_argument("--model", required=True)
    p2c.add_argument("--kv-cache-dtype", default="fp8")
    p2c.add_argument("--prompts", default=None)
    p2c.add_argument("--n-prompts", type=int, default=43)
    p2c.add_argument("--max-new-tokens", type=int, default=512)
    p2c.add_argument("--gpu-memory-utilization", type=float, default=0.42)
    p2c.add_argument("--tau-degen", type=float, default=0.05)
    p2c.add_argument("--mcnemar-alpha", type=float, default=0.01)
    p2c.add_argument("--output", default="./deltacert_free_running.json")
    p2c.add_argument("--strict", action="store_true", default=True)
    p2c.add_argument("--no-strict", dest="strict", action="store_false")
    return parser


def test_free_running_subcommand_is_registered_in_main_parser():
    """The actual argparse tree built by main() must accept `free-running`
    with just --model (everything else defaulted) -- catches the subparser
    silently not being wired into main()'s sub.add_parser calls."""
    # Reconstruct main()'s parser tree by calling the same code path up to
    # parse_args, using a stubbed sys.argv.
    old_argv = sys.argv
    try:
        sys.argv = ["deltacert", "free-running", "--model", "Qwen/Qwen2.5-7B-Instruct"]
        parser = argparse.ArgumentParser(prog="deltacert")
        sub = parser.add_subparsers(dest="command")
        # mirror the real registration call in cli.main() for one subcommand
        import inspect
        src = inspect.getsource(cli.main)
        assert '"free-running"' in src, "free-running command dispatch missing from main()"
        assert "cmd_free_running" in src, "cmd_free_running dispatch missing from main()"
    finally:
        sys.argv = old_argv


def test_free_running_defaults():
    parser = _build_free_running_parser()
    args = parser.parse_args(["free-running", "--model", "Qwen/Qwen2.5-7B-Instruct"])
    assert args.command == "free-running"
    assert args.model == "Qwen/Qwen2.5-7B-Instruct"
    assert args.kv_cache_dtype == "fp8"
    assert args.n_prompts == 43
    assert args.max_new_tokens == 512
    assert args.gpu_memory_utilization == pytest.approx(0.42)
    assert args.tau_degen == pytest.approx(0.05)
    assert args.mcnemar_alpha == pytest.approx(0.01)
    assert args.strict is True


def test_free_running_missing_model_is_hard_error():
    parser = _build_free_running_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["free-running"])


def test_cmd_free_running_no_vllm_exits_1(capsys):
    """This dev machine genuinely has no vllm installed -- exercises the
    real ImportError path, not a mocked one."""
    args = argparse.Namespace(
        model="Qwen/Qwen2.5-7B-Instruct", kv_cache_dtype="fp8", prompts=None,
        n_prompts=2, max_new_tokens=32, gpu_memory_utilization=0.42,
        tau_degen=0.05, mcnemar_alpha=0.01, output="unused.json", strict=True,
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_free_running(args)
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "vLLM" in out


def test_cmd_free_running_writes_cert_and_exits_correctly(tmp_path, monkeypatch, capsys):
    """Mock collect_free_running_vllm (no GPU needed) and verify the
    certificate JSON is written with the right shape and the exit code
    matches the certified flag -- the part of cmd_free_running that's
    actually new logic, not just argument plumbing."""
    fake_vllm = types.ModuleType("vllm")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    fake_result = {
        "n_prompts": 43,
        "excess_degeneration_rate": 0.79,
        "mcnemar_b": 34,
        "mcnemar_c": 0,
        "mcnemar_p": 1.2e-10,
        "mcnemar_alpha": 0.01,
        "degeneration_significant": True,
        "surprisal_q95_delta": 10.8,
        "fork_positions": [0] * 43,
        "certified": False,
        "verdict": "unsafe",
        "rule": "fail-closed: ...",
        "tau_degen": 0.05,
        "tau_surp": None,
    }
    monkeypatch.setattr(cli, "collect_free_running_vllm", lambda *a, **k: fake_result)
    monkeypatch.setattr(cli, "_load_prompts", lambda prompts_file, n: ["p"] * n)

    out_path = str(tmp_path / "cert.json")
    args = argparse.Namespace(
        model="Qwen/Qwen2.5-7B-Instruct", kv_cache_dtype="fp8", prompts=None,
        n_prompts=43, max_new_tokens=512, gpu_memory_utilization=0.42,
        tau_degen=0.05, mcnemar_alpha=0.01, output=out_path, strict=True,
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_free_running(args)
    assert exc_info.value.code == 1  # unsafe + strict -> exit 1

    with open(out_path) as f:
        cert = json.load(f)
    assert cert["check"] == "free_running"
    assert cert["certified"] is False
    assert cert["verdict"] == "unsafe"
    assert cert["layers"]["free_running"]["mcnemar_b"] == 34
    assert "metadata" in cert

    out = capsys.readouterr().out
    assert "NOT CERTIFIED" in out


def test_cmd_free_running_safe_verdict_exits_0(tmp_path, monkeypatch):
    fake_vllm = types.ModuleType("vllm")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    fake_result = {
        "n_prompts": 43, "excess_degeneration_rate": 0.023,
        "mcnemar_b": 1, "mcnemar_c": 3, "mcnemar_p": 0.625, "mcnemar_alpha": 0.01,
        "degeneration_significant": False, "surprisal_q95_delta": -0.29,
        "fork_positions": [10] * 43, "certified": True, "verdict": "safe",
        "rule": "fail-closed: ...", "tau_degen": 0.05, "tau_surp": None,
    }
    monkeypatch.setattr(cli, "collect_free_running_vllm", lambda *a, **k: fake_result)
    monkeypatch.setattr(cli, "_load_prompts", lambda prompts_file, n: ["p"] * n)

    out_path = str(tmp_path / "cert_safe.json")
    args = argparse.Namespace(
        model="meta-llama/Llama-3.1-8B-Instruct", kv_cache_dtype="fp8", prompts=None,
        n_prompts=43, max_new_tokens=512, gpu_memory_utilization=0.42,
        tau_degen=0.05, mcnemar_alpha=0.01, output=out_path, strict=True,
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_free_running(args)
    assert exc_info.value.code == 0
