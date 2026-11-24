# NMR-Matching Tool Agent v5 — Deployment Handoff

## 1. What v5 is

**Task**: given a candidate SMILES and an experimental NMR spectrum (13C + 1H peak list), decide Yes/No whether they match. Binary classification with a required reasoning trace.

**Base**: Qwen3-32B, full-parameter SFT on 4100 multi-turn tool-calling trajectories. Trained 2 epochs from `qwen3-32b-nmr-matching/full-exp1/checkpoint-200` (no-tool SFT-v4 baseline).

**Interface**: multi-turn Hermes tool-calling agent. The model DOES NOT answer in one shot. It emits `<think>...</think><tool_call>...</tool_call>`, the server executes the tool and returns `<tool_response>...</tool_response>`, the loop repeats until the model emits `<answer>{...}</answer>`.

**Key metric wins on `nmr_matching_test_2000.json` (2000 rows)**:
| metric | v3 no-tool SFT | v4-tool (M3) | **v5** | Gemini-3.1-Pro |
|---|---:|---:|---:|---:|
| overall acc | 72.9% | 78.3% | **84.1%** | 94.3% |
| similar_structure | 55.5% | 66.0% | **73.4%** | 96.1% |
| positive_recall | 77.8% | 76.75% | **85.4%** | ~89% |
| think-tool-consistency | n/a | 20.55% | **96.75%** | — |

The last metric is the load-bearing v5 win: v4/M3 had hallucinated `<think>` blocks (fake atom counts, made-up "6H singlet at 2.93 ppm N(CH3)2" type prose) that users could eyeball as wrong. In v5, 96.75% of trajectories reference actual tool return values by number — reasoning chain is auditable.

## 2. Model artifacts

**Location**: `/fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/saves/qwen3-32b-nmr-matching-tool/full-v5/`

**Files** (all required):
- `model-00001-of-00002.safetensors` (46.5 GB) + `model-00002-of-00002.safetensors` (14.5 GB) + `model.safetensors.index.json` — bf16 weights
- `config.json` — Qwen3ForCausalLM config
- `generation_config.json`
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` — Qwen3 tokenizer (already patched — do not swap for stock Qwen3-32B tokenizer; see caveat 8.3)

**Ancillary in the same dir** (not needed for serving; safe to skip):
- `checkpoint-500/`, `checkpoint-514/` — ZeRO-3 shards for resume-training only
- `trainer_state.json`, `trainer_log.jsonl` — training curves
- `all_results.json` — final `train_loss=0.031, eval_loss=0.1496`

## 3. Runtime dependencies (must ship with the model)

Deployment MUST include a Python-side deterministic tool runtime co-located with (or callable from) the serving process. The model emits tool calls; something has to execute them.

**Required Python files** (from `/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/`):
- `nmr_matching_runtime.py` — `NmrMatchingRuntime` class with `run_tool_call_async(tc)` method
- `named_smarts.py` — SMARTS pattern table + `resolve_pattern()` keyword resolver
- `prompt.py` — `build_system_prompt()` (canonical system prompt text)
- `prediction_cache.json` (~62 MB, 9847 entries) — precomputed NMR forward predictions keyed by canonical SMILES

**Required external Python dependencies**:
- `rdkit` (chem parsing, atom counting, SMARTS)
- The NMR forward predictor (`nmr_predictor` module at `/fs_mol/cxq/SpecMaster_jp/perturbation_generator/`) — needed ONLY for candidates missing from `prediction_cache.json`. If the deployment traffic distribution matches training pool, cache hit rate is ~95% and the remote predictor is rarely called.
- `spectral_scorer` (same dir) — pure-Python; needed for `spectral_match` scoring.

**Runtime creation** (once per request):
```python
from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime
runtime = NmrMatchingRuntime(
    record={"candidate_smiles": ..., "c_nmr_peaks": [...], "h_nmr_peaks": [...],
            "nmr_solvent": ..., "nmr_frequency": ...},
    prediction_cache=preloaded_dict,   # load prediction_cache.json once at server start
    allow_remote_predict=True,          # or False if you want cache-only, faster/deterministic
)
```

## 4. Exact prompt & tool contract

### 4.1 System prompt (verbatim from `prompt.py::build_system_prompt()`)
```
You are an expert analytical chemist. You are given a candidate molecule (SMILES) together with its experimental 13C and 1H NMR peaks. Your job is to decide whether the candidate structure MATCHES the NMR spectrum (final_answer 'Yes' if it matches, 'No' if it does not).
A positive match requires ALL peaks to be consistent with the structure; a single decisive contradiction is enough to answer 'No'.

You have access to deterministic verification tools. Do NOT estimate atom counts, spectra, or substructures in your head -- CALL the tools, they compute exact values. Gather evidence with the tools, then decide.
  * count_atoms(element='C'|'H'): exact RDKit count of carbons/hydrogens in the candidate, compared against the number of 13C signals / the summed 1H integrations. A candidate with FEWER carbons than resolved 13C signals is a hard contradiction.
  * spectral_match(): forward-predict the candidate's NMR spectrum and score its similarity (0..1, higher = better) against the experimental spectrum, with per-nucleus (13C/1H) sub-scores. Best for same-formula isomers where atom counting is uninformative.
  * check_substructure(pattern): exact RDKit check of whether a functional group / substructure is present in the candidate. `pattern` is a group NAME (e.g. 'ester', 'methoxy', 'ethoxy', 'amide', 'carbonyl', 'hydroxyl', 'nitrile', 'tms', 'aromatic_ring', 'furan', 'pyridine') or a raw SMARTS string. Use to confirm/deny a group the spectrum implies (e.g. a 9H singlet near 0 ppm implies TMS).
  * functional_group_scan(): report all known functional groups present in the candidate at once; use as a broad first pass when unsure which group to check.

Output format (STRICT):
  - Reason briefly inside <think>...</think>.
  - To use a tool, emit exactly one <tool_call>{"name": <tool>, "arguments": {...}}</tool_call>.
    You will then receive a <tool_response> with the deterministic result.
  - When you have gathered enough evidence, output your decision as:
    <answer>{"final_answer": "Yes"|"No", "confidence": <float 0..1>, "key_evidence": "<one sentence>"}</answer>
  - Call at least one tool before answering.
```

Build via `build_system_prompt()` — never hand-type; use the function so text stays in sync with training.

### 4.2 User message (JSON, exactly one turn)
```json
{
  "candidate_smiles": "CCOc1ccc(C=O)cc1",
  "c_nmr_peaks": [{"delta (ppm)": 190.7}, {"delta (ppm)": 164.1}, ...],
  "h_nmr_peaks": [{"delta": 9.85, "centroid": 9.85, "nH": 1}, ...],
  "nmr_solvent": "CDCl3",
  "nmr_frequency": 400.0
}
```
- `c_nmr_peaks`: list of `{"delta (ppm)": float}`. Empty list = no 13C data.
- `h_nmr_peaks`: list of `{"delta": float, "centroid": float, "nH": int_or_null}`. `nH` is per-peak proton integration.
- `nmr_solvent`, `nmr_frequency`: optional metadata; model does not currently use them but they are part of the trained input distribution — keep the keys even if null.

Serialize with `json.dumps(user_case, ensure_ascii=False)` and send as the `user` role content.

### 4.3 Chat template
Apply Qwen3's chat template with `tools=` argument (the 4 tool schemas — see 4.4). Do not concatenate raw text. `apply_chat_template(messages, tools=list(TOOL_SCHEMAS.values()), add_generation_prompt=True, tokenize=False)`. Training baked `<tools>` schema injection via LLaMA-Factory's `qwen3` template — inference MUST match. Failing to pass `tools=` yields off-distribution output.

### 4.4 Tool schemas (verbatim from `standalone_infer_matching.py::TOOL_SCHEMAS`)
Four OpenAI-style function tools. Serving must expose all four; the model was trained assuming full 4-tool availability.
- `count_atoms(element: "C"|"H") -> observation`
- `spectral_match(sigma_c?: number, sigma_h?: number) -> observation`
- `check_substructure(pattern: string) -> observation`
- `functional_group_scan() -> observation`

Copy the full JSON schema block from `standalone_infer_matching.py:44-107` verbatim.

### 4.5 Tool response shape (from `NmrMatchingRuntime`)
Each tool returns a JSON-string observation. Feed it back as a `tool` role message with `name=<tool_name>` and `content=<obs_json_string>`.

Example (`count_atoms(element="C")`):
```json
{"element": "C", "candidate_count": 22, "experimental_distinct_signals": 23,
 "match": false, "note": "candidate has 22 C atoms; 13C spectrum shows 23 signals. CONTRADICTION: fewer candidate carbons than resolved 13C signals."}
```
Full field lists per tool are in `nmr_matching_runtime.py::_handle_*` (L177-342).

## 5. Multi-turn loop (canonical server logic)

Pseudocode. Reference implementation: `standalone_infer_matching.py::run_case()`.

```
messages = [
    {"role": "system", "content": build_system_prompt()},
    {"role": "user",   "content": json.dumps(user_case)},
]
for turn in range(MAX_TURNS):  # cap at 8; production traces average 2-4
    prompt_text = tokenizer.apply_chat_template(
        messages, tools=list(TOOL_SCHEMAS.values()),
        add_generation_prompt=True, tokenize=False)
    output = llm.generate(prompt_text, max_new_tokens=1024,
                          temperature=0.0, stop=["</tool_call>", "</answer>"])
    text = output.text  # includes the stop token
    action = parse_action(text)  # <tool_call>{...}</tool_call> OR <answer>{...}</answer>

    messages.append({"role": "assistant", "content": text})
    if action["type"] == "answer":
        return json.loads(action["answer"])   # {"final_answer","confidence","key_evidence"}
    if action["type"] == "tool":
        obs_json = await runtime.run_tool_call_async({"function": action["call"]})
        messages.append({"role": "tool", "name": action["call"]["name"],
                         "content": obs_json["content"]})
```

## 6. Recommended generation parameters

| param | value | why |
|---|---|---|
| `temperature` | **0.0** | Deterministic tool selection; training was greedy. Non-zero degrades accuracy. |
| `top_p` | 1.0 | Only used if temperature > 0. |
| `max_new_tokens` (per turn) | 1024 | Assistant turn = `<think>...</think>` + `<tool_call>` or `<answer>`. Typical length 200-600 tokens. |
| `stop` tokens | `["</tool_call>", "</answer>"]` | Prevents runaway generation past a single action. |
| `MAX_TURNS` | 8 | Model rarely uses more than 4 tools; cap avoids infinite loops on ill-formed generations. |
| `repetition_penalty` | 1.0 | No penalty. Higher values distort tool-call JSON. |

## 7. Serving backends known to work

- **vLLM 0.8.5+** — used during training rollout eval. Works out-of-box with `--tokenizer` pointed at the v5 dir, `chat-template` = default (from tokenizer_config.json). Serve with `--served-model-name qwen3-32b-nmr-v5` and require `tools=` in every request payload.
- **HuggingFace Transformers** — reference impl in `standalone_infer_matching.py`. Slower but simpler.

Not tested: TGI, TensorRT-LLM. Tool-format template compatibility unknown.

**Hardware**: full-precision inference needs ~62 GB. Single A100-80G or 2×A100-40G (tensor parallel) is comfortable. bf16 KV cache; typical multi-turn peak KV ~2-4 GB.

## 8. Known caveats

### 8.1 The `<tools>` schema must be attached at inference
The Qwen3 template auto-injects `<tools>...</tools>` when `tools=` is passed to `apply_chat_template`. Training was done via LLaMA-Factory with `function_tag: function_call`, `observation_tag: observation`, `qwen3` template — same injection. If deployment forgets `tools=`, the model still generates but tool-call format degrades. Enforce this at the request-parsing layer.

### 8.2 `spectral_match` cache miss handling
The 9847-entry `prediction_cache.json` covers the training + test SMILES set. In deployment, novel candidates will miss the cache. Options:
- (fast) `allow_remote_predict=False`: cache-miss returns `similarity=0, ok=false, note="no predicted spectrum available"`. Model has been trained on `ok=false` observations and gracefully proceeds with other tools.
- (accurate) `allow_remote_predict=True`: falls back to the Uni-Mol NMR predictor. Adds ~1-3s latency per unique novel SMILES. Cache the result to disk to reduce cold-hit cost. Predictor endpoint config lives inside `nmr_predictor.py::_http_predict_one` (edit if service URL changes).

### 8.3 Tokenizer patch — do not overwrite
The v5 tokenizer at `saves/.../full-v5/tokenizer.json` is a patched Qwen3 tokenizer where `extra_special_tokens` was serialized as an empty dict `{}` instead of a list (fixes a compat error with older transformers; see memory `feedback_llamafactory_tokenizer_extraspec`). If you re-download the stock Qwen3-32B tokenizer and drop it in, older transformers versions will crash on `.keys()`. Keep the shipped tokenizer.

### 8.4 Accuracy gaps
- On 6 tiny functional_group_replacement subclasses (alcohol_to_thiol / amide_to_ester / ester_to_amide × {perturb, hard_nmr}), v5 acc is 37.5% - 75%. v5.1 top-up is data-ready and yaml-ready (see `qwen3_32b_nmr_matching_tool_v5_1.yaml`) — train it if these subclasses matter for the traffic.
- Positive_recall 85.4% (vs 92%+ target). "positive" here means the candidate IS the ground truth. False-No rate ~15% — v5 sometimes rejects legitimate matches when peak counts are noisy. If false-No is costly, consider a confidence-threshold post-processor or run RL (M4 plan, not in this deployment).

### 8.5 Not a general-purpose chat model
v5 has been fine-tuned on a narrow task. It will still respond to arbitrary chat, but degradation on out-of-distribution prompts is significant. Do not serve as a general chatbot. Prefix the serving policy with input validation: expect the `user` message to be a JSON object with `candidate_smiles` + peak lists; reject other shapes at the gateway.

## 9. Health-check + smoke tests

**Fastest smoke** — one row from the test set, expect `<answer>` in <5 turns:
```bash
cd /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL
python recipe/nmr_matching_agent/standalone_infer_matching.py \
  --model_path /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/saves/qwen3-32b-nmr-matching-tool/full-v5 \
  --from_test /fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json \
  --n 3 --output /tmp/v5_smoke.json
```

**Full regression** — full test_2000, target overall ≥84%, think-tool-consistency ≥95%:
```bash
python recipe/nmr_matching_agent/standalone_infer_matching.py \
  --model_path /fs_mol/cxq/.../full-v5 \
  --from_test /fs_mol/cxq/SpecMaster_jp/data/nmr_matching_test_2000.json --n 2000 \
  --output /tmp/v5_full.json
python /fs_mol/cxq/SpecMaster_jp/scripts/eval_think_consistency.py --in /tmp/v5_full.json
```

Baseline numbers for regression alerts:
- overall acc: 84.1% ± 1pt
- similar_structure: 73.4% ± 2pt
- positive_recall: 85.4% ± 2pt
- think-tool-consistency: 96.75% ± 1pt

If numbers drift below these, first check: (a) `tools=` is being passed at inference, (b) `temperature=0`, (c) `prediction_cache.json` is loaded (not empty), (d) tokenizer is the shipped patched version (not stock Qwen3).

## 10. Contact / provenance

- Training data provenance: `recipe/nmr_matching_agent/data_v5/tool_sft_v5.jsonl` (4100 rows, Hermes messages format). Half from Path B rewriter of v4 CoT + half from Path A closed-model teacher distillation (Gemini-3.1-Pro / GPT-5.5 / DeepSeek-V4-Pro).
- Training recipe: `examples/train_full/qwen3_32b_nmr_matching_tool_v5.yaml`.
- Rollout reference: `standalone_infer_matching.py`.
- Plan file (design rationale): `/root/.claude/plans/gemini3-1-pro-gpt5-5-deepseek-v4-pro-sf-shiny-shell.md`.
