"""Entry tests for the step-01 review fixes.

R1 drives standalone_infer_matching.main. R2 drives the production split script,
then loads each shard file and merges through that script's merge command.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from recipe.nmr_matching_agent import standalone_infer_matching as infer

SPLIT_PATH = Path("/fs_mol/cxq/SpecMaster_jp/scripts/_split_and_merge_infer.py")
TEXT = (
    "Candidate SMILES: CCO\n"
    "13C NMR (400 MHz, CDCl3): 57.8, 18.2\n"
    "1H NMR (400 MHz, CDCl3): 1.18 (t, 3H)\n"
)


def _load_split_module():
    spec = importlib.util.spec_from_file_location("split_and_merge_infer_step01", SPLIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def block_model_network_and_cache(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise RuntimeError("blocked load")

    monkeypatch.setattr(infer, "load_model_and_tokenizer", blocked)
    monkeypatch.setattr(infer, "load_tools", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    import recipe.nmr_matching_agent.rule_based_router as router

    monkeypatch.setattr(router, "load_prediction_cache", lambda path=None: {})
    monkeypatch.setattr(router, "load_config", lambda path=None: {
        "sim_threshold_both": 0.85,
        "sim_threshold_strict": 0.92,
        "h_tolerance": 2,
        "template_variant_seed": 0,
        "template_languages": ["en"],
    })


def _forbidden(payload) -> None:
    encoded = json.dumps(payload, ensure_ascii=False)
    for key in (
        "_label", "_negative_type", "_gt_smiles", "_candidate_smiles",
        "_harness_pair_id", "_harness_run_id", "_pair_id", "pair_id",
        "teacher_pred", "output",
    ):
        assert key not in encoded


def _assert_model_message(case) -> None:
    messages = infer.make_messages(case)
    user = json.loads(next(message["content"] for message in messages if message["role"] == "user"))
    assert user == case
    _forbidden(user)


class _Router:
    seen = []

    def __init__(self, case, prediction_cache=None, config=None, allow_remote_predict=True):
        self.case = copy.deepcopy(case)
        _Router.seen.append(self.case)
        self.template_variant_seed = 0

    def analyze(self):
        return SimpleNamespace(
            route="model",
            decision=SimpleNamespace(
                assignment_vetoed=False,
                assignment_veto_reasons=[],
                assignment_shadow_mode=True,
            ),
        )


def _install_fakes(monkeypatch):
    seen = []

    async def agent(_tok, _model, case, max_turns=6, tools=None):
        seen.append(copy.deepcopy(case))
        shift = case["c_nmr_peaks"][0]["delta (ppm)"]
        return {
            "answer": f"shift={shift}",
            "fail_reason": "forced" if shift == 30.0 else None,
            "tool_sequence": [],
            "transcript": [],
            "case_candidate": case.get("candidate_smiles"),
            "_label": "from-model",
            "_negative_type": "from-model",
        }

    monkeypatch.setattr(infer, "run_agent", agent)
    monkeypatch.setattr("recipe.nmr_matching_agent.rule_based_router.RuleBasedRouter", _Router)
    monkeypatch.setattr(infer, "load_model_and_tokenizer", lambda _path: ("tok", "model"))
    _Router.seen = []
    return seen


def _run_main(monkeypatch, path: Path, output: Path, n: int):
    monkeypatch.setattr(sys, "argv", [
        "standalone_infer_matching.py",
        "--model_path", "fake-model",
        "--from_test", str(path),
        "--n", str(n),
        "--output", str(output),
        "--use_router",
    ])
    infer.main()


def test_r1_main_keeps_text_spectrum_when_candidate_is_repeated(tmp_path, monkeypatch):
    seen = _install_fakes(monkeypatch)
    records = [
        {"input": TEXT, "_label": 1, "_negative_type": "positive", "_gt_smiles": "CCO", "_candidate_smiles": "NOT-USED"},
        {"input": TEXT, "candidate_smiles": "CCO", "_label": 0, "_negative_type": "neg/shift", "_gt_smiles": "CC"},
        {"input": TEXT, "candidate": "CCO", "_label": 1, "_negative_type": "positive"},
        {
            "candidate_smiles": "CCC",
            "candidate": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 11.0, "label": 1}],
            "h_nmr_peaks": [{"delta": 1.2, "centroid": 1.2, "nH": None, "category": "No", "gt_smiles": "C"}],
            "experimental_total_nH": 0,
            "solvent": "DMSO",
            "_label": 0,
            "_negative_type": "neg/random",
        },
        {
            "input": TEXT,
            "candidate_smiles": "CCO",
            "h_nmr_peaks": [{"delta": 1.18, "centroid": 1.18, "nH": 3, "category": "t"}],
            "_label": 1,
            "_negative_type": "positive",
        },
    ]
    source = tmp_path / "mixed.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    output = tmp_path / "out.json"
    _run_main(monkeypatch, source, output, len(records))

    assert len(seen) == 5
    assert seen == _Router.seen
    for case in seen:
        _forbidden(case)
        _assert_model_message(case)
    assert seen[0] == seen[1] == seen[2]
    assert len(seen[0]["c_nmr_peaks"]) == 2
    assert len(seen[0]["h_nmr_peaks"]) == 1
    assert seen[0]["h_nmr_peaks"][0]["nH"] == 3
    assert seen[0]["candidate_smiles"] == "CCO"
    assert seen[3]["candidate_smiles"] == "CCC"
    assert seen[3]["c_nmr_peaks"] == [{"delta (ppm)": 11.0}]
    assert seen[3]["h_nmr_peaks"][0]["nH"] is None
    assert seen[3]["h_nmr_peaks"][0]["category"] == "No"
    assert seen[3]["experimental_total_nH"] == 0
    assert seen[3]["nmr_solvent"] == "DMSO"
    assert seen[4]["c_nmr_peaks"] == seen[0]["c_nmr_peaks"]
    assert seen[4]["h_nmr_peaks"][0]["category"] == "t"
    assert seen[4]["h_nmr_peaks"][0]["nH"] == 3

    rows = json.loads(output.read_text(encoding="utf-8"))
    assert [row["_label"] for row in rows] == [1, 0, 1, 0, 1]
    assert [row["_negative_type"] for row in rows] == [
        "positive", "neg/shift", "positive", "neg/random", "positive",
    ]
    assert rows[0]["_candidate_smiles"] == "NOT-USED"
    assert rows[0]["case_candidate"] == "CCO"
    assert all(row["_label"] != "from-model" for row in rows)


def test_r1_main_rejects_conflicts_and_empty_spectra(tmp_path, monkeypatch):
    seen = _install_fakes(monkeypatch)

    def expect_error(record, message):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([record]), encoding="utf-8")
        output = tmp_path / "should-not-exist.json"
        with pytest.raises(ValueError, match=message):
            _run_main(monkeypatch, path, output, 1)
        assert not output.exists()

    expect_error(
        {"input": TEXT, "candidate_smiles": "CCC", "_label": 1, "_gt_smiles": "CCO"},
        "candidate SMILES differs",
    )
    expect_error(
        {"input": TEXT, "c_nmr_peaks": [{"delta (ppm)": 1.0}], "_label": 0},
        "13C peak count differs",
    )
    expect_error(
        {"candidate_smiles": "CCO", "c_nmr_peaks": [], "h_nmr_peaks": []},
        "no experimental NMR peaks",
    )
    assert seen == []
    assert _Router.seen == []


def test_r2_production_split_load_and_merge_restores_global_order(tmp_path, monkeypatch):
    seen = _install_fakes(monkeypatch)
    records = [
        {
            "case_id": "a", "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 57.8}], "h_nmr_peaks": [],
            "_label": 1, "_negative_type": "positive", "_gt_smiles": "CCO", "output": "Yes",
        },
        {
            "case_id": "b", "candidate_smiles": "CCO",
            "c_nmr_peaks": [{"delta (ppm)": 18.2}], "h_nmr_peaks": [],
            "_label": 0, "_negative_type": "neg/shift", "_gt_smiles": "CC", "output": "No",
        },
        {
            "candidate_smiles": "CCC",
            "c_nmr_peaks": [{"delta (ppm)": 11.0}], "h_nmr_peaks": [],
            "_label": 0, "_negative_type": "neg/random", "_gt_smiles": "CCO",
        },
        {
            "idx": "dup", "candidate_smiles": "CC",
            "c_nmr_peaks": [{"delta (ppm)": 5.0}], "h_nmr_peaks": [],
            "_label": 1, "_negative_type": "positive", "_gt_smiles": "CC",
        },
        {
            "idx": "dup", "candidate_smiles": "CC",
            "c_nmr_peaks": [{"delta (ppm)": 30.0}], "h_nmr_peaks": [],
            "_label": 0, "_negative_type": "neg/decoy", "_gt_smiles": "CCC",
        },
    ]
    source = tmp_path / "full.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    source_hash = source.read_bytes()
    outdir = tmp_path / "shards"
    splitter = _load_split_module()
    monkeypatch.setattr(sys, "argv", [
        "_split_and_merge_infer.py", "split",
        "--input", str(source), "--n", "2", "--outdir", str(outdir),
    ])
    splitter.main()
    assert source.read_bytes() == source_hash

    manifest = json.loads((outdir / "shard_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_rows"] == 5
    assert manifest["pair_ids"] == [0, 1, 2, 3, 4]
    shard_paths = [outdir / "shard_0.json", outdir / "shard_1.json"]
    loaded = [infer.load_from_test(str(path), None) for path in shard_paths]
    assert [[item["pair_id"] for item in shard] for shard in loaded] == [[0, 1, 2], [3, 4]]
    assert len({item["run_id"] for shard in loaded for item in shard}) == 1
    for shard in loaded:
        for item in shard:
            _forbidden(item["inference_input"])

    result_paths = []
    for index, path in enumerate(shard_paths):
        output = tmp_path / f"result_{index}.json"
        _run_main(monkeypatch, path, output, 10)
        rows = json.loads(output.read_text(encoding="utf-8"))
        rows.reverse()
        output.write_text(json.dumps(rows), encoding="utf-8")
        result_paths.append(output)

    merged_path = tmp_path / "merged.json"
    monkeypatch.setattr(sys, "argv", [
        "_split_and_merge_infer.py", "merge",
        "--inputs", str(result_paths[1]), str(result_paths[0]),
        "--manifest", str(outdir / "shard_manifest.json"),
        "--output", str(merged_path),
    ])
    splitter.main()
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    assert len(merged) == 5
    assert [row["_pair_id"] for row in merged] == [0, 1, 2, 3, 4]
    assert len({row["_pair_id"] for row in merged}) == 5
    assert [row["answer"] for row in merged] == [
        "shift=57.8", "shift=18.2", "shift=11.0", "shift=5.0", "shift=30.0",
    ]
    assert [row["_label"] for row in merged] == [1, 0, 0, 1, 0]
    assert [row["_negative_type"] for row in merged] == [
        "positive", "neg/shift", "neg/random", "positive", "neg/decoy",
    ]
    assert [row["_sample_id"] for row in merged] == ["a", "b", None, "dup", "dup"]
    assert [row["_gt_smiles"] for row in merged] == ["CCO", "CC", "CCO", "CC", "CCC"]
    assert merged[0]["case_candidate"] == merged[1]["case_candidate"] == "CCO"
    assert merged[0]["_label"] != merged[1]["_label"]
    assert merged[4]["fail_reason"] == "forced"
    assert len(seen) == 5
    assert seen == _Router.seen
    for case in seen:
        _forbidden(case)
        _assert_model_message(case)


def test_r2_merge_rejects_duplicates_missing_rows_and_mixed_runs(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    source = tmp_path / "full.json"
    source.write_text(json.dumps([
        {"candidate_smiles": "CCO", "c_nmr_peaks": [{"delta (ppm)": 57.8}], "h_nmr_peaks": [], "_label": 1},
        {"candidate_smiles": "CC", "c_nmr_peaks": [{"delta (ppm)": 18.2}], "h_nmr_peaks": [], "_label": 0},
    ]), encoding="utf-8")
    outdir = tmp_path / "shards"
    splitter = _load_split_module()
    monkeypatch.setattr(sys, "argv", [
        "_split_and_merge_infer.py", "split",
        "--input", str(source), "--n", "2", "--outdir", str(outdir),
    ])
    splitter.main()
    outputs = []
    for index in (0, 1):
        output = tmp_path / f"result_{index}.json"
        _run_main(monkeypatch, outdir / f"shard_{index}.json", output, 10)
        outputs.append(output)
    manifest = str(outdir / "shard_manifest.json")
    rows = []
    for path in outputs:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))

    def merge_rows(bad_rows):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad_rows), encoding="utf-8")
        dest = tmp_path / "bad-out.json"
        monkeypatch.setattr(sys, "argv", [
            "_split_and_merge_infer.py", "merge",
            "--inputs", str(path), "--manifest", manifest, "--output", str(dest),
        ])
        splitter.main()

    with pytest.raises(ValueError, match="duplicate"):
        merge_rows(rows + [dict(rows[0])])
    with pytest.raises(ValueError, match="missing"):
        merge_rows(rows[:-1])
    mixed = [dict(row) for row in rows]
    mixed[1]["_harness_run_id"] = "other-run"
    with pytest.raises(ValueError, match="different evaluation runs"):
        merge_rows(mixed)
    unstamped = [{"answer": row["answer"], "_label": row["_label"]} for row in rows]
    with pytest.raises(ValueError, match="no harness pair ids"):
        merge_rows(unstamped)


def test_r2_legacy_results_keep_input_order_without_invented_ids(tmp_path, capsys):
    splitter = _load_split_module()
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps([{"answer": "second-shard", "_label": 0}]), encoding="utf-8")
    second.write_text(json.dumps([{"answer": "first-shard", "_label": 1}]), encoding="utf-8")
    output = tmp_path / "legacy.json"
    splitter.cmd_merge(SimpleNamespace(
        inputs=[str(first), str(second)],
        manifest=None,
        output=str(output),
    ))
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert [row["answer"] for row in rows] == ["second-shard", "first-shard"]
    assert all("_pair_id" not in row and "_harness_run_id" not in row for row in rows)
    assert "not verified" in capsys.readouterr().err
