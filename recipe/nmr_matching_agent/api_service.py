"""Single-GPU FastAPI service for the NMR matching router + tool agent."""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from recipe.nmr_matching_agent.rule_based_router import (
    RuleBasedRouter,
    build_prefilled_messages,
    load_config,
)
from recipe.nmr_matching_agent.prompt import build_system_prompt
from recipe.nmr_matching_agent.standalone_infer_matching import (
    ANSWER_RE,
    build_template_transcript,
    build_user_case,
    generate_once,
    load_model_and_tokenizer,
    load_tools,
    run_agent,
)

THINK_RE = re.compile(r"<think>\s*([\s\S]*?)\s*</think>", re.I)
DEFAULT_MODEL = "/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/saves/qwen3-32b-nmr-matching-tool/full-v6_4/checkpoint-63"


class MatchRequest(BaseModel):
    candidate_smiles: str = Field(min_length=1)
    c_nmr_peaks: List[Dict[str, Any]] = Field(default_factory=list)
    h_nmr_peaks: List[Dict[str, Any]] = Field(default_factory=list)
    nmr_solvent: Optional[str] = None
    nmr_frequency: Optional[str] = None
    experimental_total_nH: Optional[float] = None
    include_model_reasoning_in_key_evidence: bool = False

    @model_validator(mode="after")
    def require_spectrum(self):
        if not self.c_nmr_peaks and not self.h_nmr_peaks:
            raise ValueError("at least one of c_nmr_peaks or h_nmr_peaks is required")
        return self


class ToolCallPublic(BaseModel):
    name: str
    arguments: Dict[str, Any]


class TrajectoryTurnPublic(BaseModel):
    turn: int
    think: str
    tool_call: Optional[ToolCallPublic] = None
    tool_output: Optional[Dict[str, Any] | str] = None
    final_answer: Optional[Literal["Yes", "No"]] = None


class MatchResponse(BaseModel):
    final_answer: Literal["Yes", "No"]
    final_think: str
    key_evidence: str
    trajectory: List[TrajectoryTurnPublic]


def _think(text: str) -> str:
    match = THINK_RE.search(text or "")
    return match.group(1).strip() if match else ""


def _answer_object(tagged: str) -> Dict[str, Any]:
    match = ANSWER_RE.search(tagged or "")
    if not match:
        raise ValueError("final answer block missing")
    obj = json.loads(match.group(1))
    if not isinstance(obj, dict) or obj.get("final_answer") not in {"Yes", "No"}:
        raise ValueError("invalid final answer object")
    return obj


def _model_reasoning(result: Dict[str, Any]) -> str:
    for row in reversed(result.get("transcript") or []):
        if (row.get("action") or {}).get("type") == "answer":
            reasoning = _think(row.get("assistant", ""))
            return re.sub(
                r"\s*Therefore\s+final_answer\s*=\s*(?:Yes|No)\.?\s*$",
                "",
                reasoning,
                flags=re.I,
            ).strip()
    return ""


def _prepend_model_reasoning(tagged: str, reasoning: str) -> str:
    if not reasoning:
        raise ValueError("model reasoning is unavailable")
    obj = _answer_object(tagged)
    deterministic = str(obj.get("key_evidence") or "").strip()
    if obj.get("final_answer") == "No":
        # A negative decision must expose exactly one deterministic, tool-grounded
        # contradiction.  Raw model prose remains available in the trajectory and
        # must not dilute or contradict the public key_evidence field.
        obj["key_evidence"] = deterministic
    else:
        obj["key_evidence"] = f"{reasoning} {deterministic}".strip()
    think = _think(tagged)
    return (
        f"<think>{think}</think>\n"
        f"<answer>{json.dumps(obj, ensure_ascii=False)}</answer>"
    )


def public_response(result: Dict[str, Any]) -> MatchResponse:
    """Remove confidence and route/template metadata while retaining requested evidence."""
    answer = _answer_object(str(result.get("answer") or ""))
    final = answer["final_answer"]
    turns: List[TrajectoryTurnPublic] = []
    for row in result.get("transcript") or []:
        action = row.get("action") or {}
        item: Dict[str, Any] = {"turn": int(row.get("turn", len(turns) + 1)), "think": _think(row.get("assistant", ""))}
        if action.get("type") == "tool":
            raw = row.get("tool_observation")
            try:
                output = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                output = str(raw)
            item["tool_call"] = {"name": action.get("name"), "arguments": action.get("arguments") or {}}
            item["tool_output"] = output
        elif action.get("type") == "answer":
            item["final_answer"] = final
        turns.append(TrajectoryTurnPublic.model_validate(item))
    return MatchResponse(
        final_answer=final,
        final_think=_think(str(result["answer"])),
        key_evidence=str(answer.get("key_evidence") or ""),
        trajectory=turns,
    )


def _prefilled_model_reasoning(tokenizer, model, case: Dict[str, Any], routed) -> str:
    """One generation over Router observations; tools are not exposed or executed."""
    messages = build_prefilled_messages(case, routed, build_system_prompt())
    raw = generate_once(tokenizer, model, messages, use_tools=False)
    if not ANSWER_RE.search(raw):
        raise RuntimeError("prefilled model did not produce a final answer")
    reasoning = _think(raw)
    if not reasoning:
        raise RuntimeError("prefilled model did not produce a think block")
    return re.sub(
        r"\s*Therefore\s+final_answer\s*=\s*(?:Yes|No)\.?\s*$",
        "",
        reasoning,
        flags=re.I,
    ).strip()


class ServiceState:
    def __init__(self) -> None:
        self.model_path = os.getenv("NMR_MATCHING_MODEL_PATH", DEFAULT_MODEL)
        self.max_turns = int(os.getenv("NMR_MATCHING_MAX_TURNS", "6"))
        self.allow_remote = os.getenv("NMR_MATCHING_ALLOW_REMOTE_PREDICT", "1") not in {"0", "false", "False"}
        self.tokenizer = None
        self.model = None
        self.tools = None
        self.router_cache = None
        self.router_config = None
        self.ready = False
        self.lock = threading.Lock()

    def load(self) -> None:
        cfg = load_config()
        cfg["sim_threshold_both"] = float(os.getenv("NMR_ROUTER_SIM_BOTH", "0.85"))
        cfg["sim_threshold_strict"] = float(os.getenv("NMR_ROUTER_SIM_STRICT", "0.92"))
        cfg["h_tolerance"] = int(os.getenv("NMR_ROUTER_H_TOLERANCE", "2"))
        cfg["template_variant_seed"] = int(os.getenv("NMR_ROUTER_TEMPLATE_SEED", "0"))
        self.router_config = cfg
        self.tools = load_tools(
            allow_remote_predict=self.allow_remote,
            h_tolerance=cfg["h_tolerance"],
        )
        self.router_cache = next(iter(self.tools.values())).session_manager.prediction_cache
        self.tokenizer, self.model = load_model_and_tokenizer(self.model_path)
        self.ready = True

    def infer(self, payload: MatchRequest) -> MatchResponse:
        if not self.ready:
            raise RuntimeError("service is not ready")
        case = build_user_case(payload.model_dump())
        with self.lock:  # one 32B generation stream at a time on an 80G card
            router = RuleBasedRouter(
                case,
                prediction_cache=self.router_cache,
                config=self.router_config,
                allow_remote_predict=self.allow_remote,
            )
            routed = router.analyze()
            def run_model_path() -> Dict[str, Any]:
                return asyncio.run(
                    run_agent(
                        self.tokenizer,
                        self.model,
                        case,
                        max_turns=self.max_turns,
                        tools=self.tools,
                    )
                )

            if routed.route == "template":
                transcript = build_template_transcript(routed, routed.full_assistant_turn)
                result = {"answer": routed.full_assistant_turn, "transcript": transcript, "fail_reason": None}
                model_reasoning = (
                    _prefilled_model_reasoning(self.tokenizer, self.model, case, routed)
                    if payload.include_model_reasoning_in_key_evidence
                    else ""
                )
            else:
                result = run_model_path()
                model_reasoning = _model_reasoning(result)

            if result.get("fail_reason") or not result.get("answer"):
                raise RuntimeError(f"inference failed: {result.get('fail_reason') or 'empty_answer'}")
            if payload.include_model_reasoning_in_key_evidence:
                result["answer"] = _prepend_model_reasoning(result["answer"], model_reasoning)
                if routed.route == "template":
                    answer_body = ANSWER_RE.search(result["answer"])
                    result["transcript"][-1]["assistant"] = result["answer"]
                    result["transcript"][-1]["action"]["answer"] = answer_body.group(1).strip()
            return public_response(result)


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("NMR_MATCHING_SKIP_STARTUP_LOAD") != "1":
        state.load()
    yield


app = FastAPI(title="NMR Matching API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ready": state.ready}


@app.post("/match", response_model=MatchResponse, response_model_exclude_none=True)
def match(payload: MatchRequest) -> MatchResponse:
    try:
        return state.infer(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
