"""VeRL tool wrapper for the NMR spectrum<->SMILES MATCHING agent.

Sibling of `verl.tools.nmr_tools_with_formula`, but for the *matching* (verify-given-candidate)
task rather than the *generation* (search-a-pool) task. Key differences:
  * The user record carries `candidate_smiles` + experimental peaks; there is NO molecular
    formula requirement and NO candidate pool.
  * Backed by `recipe/nmr_matching_agent/nmr_matching_runtime.NmrMatchingRuntime`, which exposes
    deterministic evidence tools (count_atoms, spectral_match, check_substructure,
    functional_group_scan).
  * No remote repair/rank service. The only slow op (NMR forward prediction for spectral_match)
    is served from a precomputed cache loaded once by the session manager.

All tool entries in `tool_config_matching.yaml` share one `NmrMatchingSessionManager` singleton.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op


def _load_module_from_path(module_name: str, py_path: str):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from path: {py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _extract_first_user_content(raw_prompt: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    if not isinstance(raw_prompt, list):
        return None
    for m in raw_prompt:
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
    return None


def _build_record_from_user_json(user_obj: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Build a matching-runtime record. Returns (record, error_message_or_None)."""
    cand = user_obj.get("candidate_smiles") or user_obj.get("candidate")
    if isinstance(user_obj.get("input"), dict) and not cand:
        cand = user_obj["input"].get("candidate_smiles")
    if not isinstance(cand, str) or not cand.strip():
        return {}, "missing candidate_smiles in user JSON"

    h_peaks = user_obj.get("h_nmr_peaks") if isinstance(user_obj.get("h_nmr_peaks"), list) else []
    c_peaks = user_obj.get("c_nmr_peaks") if isinstance(user_obj.get("c_nmr_peaks"), list) else []
    if not h_peaks and not c_peaks:
        return {}, "record has no NMR peaks"

    rec: Dict[str, Any] = {
        "idx": user_obj.get("idx", -1),
        "candidate_smiles": cand.strip(),
        "h_nmr_peaks": h_peaks,
        "c_nmr_peaks": c_peaks,
        "nmr_solvent": user_obj.get("nmr_solvent"),
        "nmr_frequency": user_obj.get("nmr_frequency"),
        "experimental_total_nH": user_obj.get("experimental_total_nH"),
    }
    return rec, None


@dataclass
class _Session:
    session_key: str
    runtime: Any
    refcnt: int = 0
    meta: dict = field(default_factory=dict)


class NmrMatchingSessionManager:
    def __init__(
        self,
        *,
        prediction_cache_path: Optional[str] = None,
        allow_remote_predict: bool = False,
        sigma_c: float = 5.0,
        sigma_h: float = 0.3,
        h_tolerance: int = 2,
        runtime_module_path: Optional[str] = None,
        runtime_class_name: str = "NmrMatchingRuntime",
    ) -> None:
        self.allow_remote_predict = bool(allow_remote_predict)
        self.sigma_c = float(sigma_c)
        self.sigma_h = float(sigma_h)
        self.h_tolerance = int(h_tolerance)
        self.sessions: Dict[str, _Session] = {}

        # Load the runtime module by path (recipe dir is not an importable package).
        repo_root = Path(__file__).resolve().parents[2]
        runtime_py = (
            Path(runtime_module_path).expanduser().resolve()
            if runtime_module_path
            else repo_root / "recipe" / "nmr_matching_agent" / "nmr_matching_runtime.py"
        )
        if not runtime_py.exists():
            raise RuntimeError(f"matching runtime not found at: {runtime_py}")
        runtime_mod = _load_module_from_path("nmr_matching_runtime_for_rl", str(runtime_py))
        self.runtime_cls = getattr(runtime_mod, runtime_class_name)

        # Load the prediction cache once (canonical-SMILES keyed) so rollout makes no network call.
        self.prediction_cache: Dict[str, Dict[str, Any]] = {}
        if prediction_cache_path and os.path.exists(prediction_cache_path):
            try:
                with open(prediction_cache_path) as f:
                    self.prediction_cache = json.load(f)
                print(
                    f"[NmrMatchingSessionManager] loaded {len(self.prediction_cache)} "
                    f"cached predictions from {prediction_cache_path}"
                )
            except Exception as e:  # pragma: no cover - defensive
                print(f"[NmrMatchingSessionManager][WARN] failed to load cache: {e}")

    def create_or_get_session(self, session_key: str, record: Dict[str, Any]) -> str:
        if session_key in self.sessions:
            self.sessions[session_key].refcnt += 1
            return session_key
        runtime = self.runtime_cls(
            record=record,
            prediction_cache=self.prediction_cache,
            allow_remote_predict=self.allow_remote_predict,
            sigma_c=self.sigma_c,
            sigma_h=self.sigma_h,
            h_tolerance=self.h_tolerance,
        )
        self.sessions[session_key] = _Session(session_key=session_key, runtime=runtime, refcnt=1)
        return session_key

    def get_session(self, session_key: str) -> _Session:
        return self.sessions[session_key]

    def release_session(self, session_key: str) -> None:
        sess = self.sessions.get(session_key)
        if not sess:
            return
        sess.refcnt = max(sess.refcnt - 1, 0)
        if sess.refcnt <= 0:
            self.sessions.pop(session_key, None)


_GLOBAL_NMR_MATCHING_SESSION_MANAGER: Optional[NmrMatchingSessionManager] = None


class NmrMatchingTool(BaseTool):
    """One VeRL tool entry per evidence tool; all entries share the session-manager singleton."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        global _GLOBAL_NMR_MATCHING_SESSION_MANAGER

        maybe_manager = config.get("session_manager")
        if maybe_manager is not None:
            self.session_manager: NmrMatchingSessionManager = maybe_manager
        else:
            if _GLOBAL_NMR_MATCHING_SESSION_MANAGER is None:
                _GLOBAL_NMR_MATCHING_SESSION_MANAGER = NmrMatchingSessionManager(
                    prediction_cache_path=config.get("prediction_cache_path"),
                    allow_remote_predict=bool(config.get("allow_remote_predict", False)),
                    sigma_c=float(config.get("sigma_c", 5.0)),
                    sigma_h=float(config.get("sigma_h", 0.3)),
                    h_tolerance=int(config.get("h_tolerance", 2)),
                    runtime_module_path=config.get("runtime_module_path"),
                    runtime_class_name=str(config.get("runtime_class_name", "NmrMatchingRuntime")),
                )
            self.session_manager = _GLOBAL_NMR_MATCHING_SESSION_MANAGER

        self._instance_session_key: Dict[str, str] = {}

    def _resolve_session_key(self, instance_id: str, create_kwargs: Dict[str, Any]) -> str:
        for k in ("episode_id", "conversation_id", "trajectory_id", "session_id"):
            v = create_kwargs.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return instance_id

    async def create(
        self,
        instance_id: Optional[str] = None,
        create_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = dict(create_kwargs or {})
        if "raw_prompt" in kwargs and "raw_prompt" not in create_kwargs:
            create_kwargs["raw_prompt"] = kwargs["raw_prompt"]

        raw_prompt = create_kwargs.get("raw_prompt")
        user_text = _extract_first_user_content(raw_prompt)
        if not user_text:
            return instance_id, ToolResponse(text="[NmrMatchingTool] Missing user JSON in raw_prompt.")
        try:
            user_obj = json.loads(user_text)
        except Exception:
            return instance_id, ToolResponse(text="[NmrMatchingTool] user content is not valid JSON.")
        if not isinstance(user_obj, dict):
            return instance_id, ToolResponse(text="[NmrMatchingTool] user JSON is not an object.")

        record, err = _build_record_from_user_json(user_obj)
        if err is not None:
            return instance_id, ToolResponse(text=f"[NmrMatchingTool] {err}")

        session_key = self._resolve_session_key(instance_id, create_kwargs)
        self.session_manager.create_or_get_session(session_key=session_key, record=record)
        self._instance_session_key[instance_id] = session_key
        return instance_id, ToolResponse(
            text=f"[NmrMatchingTool] Session {session_key} ready. candidate={record.get('candidate_smiles')!r}"
        )

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> Tuple[ToolResponse, float, dict]:
        session_key = self._instance_session_key.get(instance_id)
        if session_key is None:
            return ToolResponse(text="[NmrMatchingTool] session not initialized"), 0.0, {"ok": False}
        sess = self.session_manager.get_session(session_key)
        runtime = sess.runtime

        tool_call = {
            "id": f"call_{uuid4().hex}",
            "type": "function",
            "function": {"name": self.name, "arguments": parameters or {}},
        }
        try:
            result = await runtime.run_tool_call_async(tool_call)
        except Exception as e:
            return (
                ToolResponse(text=f"[NmrMatchingTool] runtime error: {type(e).__name__}: {e}"),
                0.0,
                {"ok": False},
            )

        obs_text = result.get("content", "") if isinstance(result, dict) else ""
        extra = result.get("extra", {}) if isinstance(result, dict) else {}
        ok = bool(result.get("ok", True)) if isinstance(result, dict) else False
        metrics = {"ok": ok, **{f"extra_{k}": v for k, v in (extra or {}).items()}}
        return ToolResponse(text=obs_text), 0.0, metrics

    async def release(self, instance_id: str, **kwargs) -> None:
        session_key = self._instance_session_key.pop(instance_id, None)
        if session_key is None:
            return
        self.session_manager.release_session(session_key)
