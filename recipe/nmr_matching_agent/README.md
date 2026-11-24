# NMR 谱图-SMILES 匹配：多轮工具调用 Agent

把匹配判别外包给**确定性工具**，绕过 SFT/无工具 RL 的 ~73% 判别力天花板。候选 SMILES 是给定的，
工具是「验证给定候选」而非「搜索候选池」（比 `recipe/nmr/` 的 generate/search/optimize/score 简单）。

计划全文见 `/root/.claude/plans/fs-mol-cxq-specmaster-jp-agent-tool-rou-hashed-sunbeam.md`。

## 工具集（细粒度 4 工具 + answer 伪工具）

| 工具 | 参数 | 对应 CoT action | 状态 |
|------|------|-----------------|------|
| `count_atoms` | `{element:"C"\|"H"}` | count_carbons/count_hydrogens | ✅ M0 |
| `spectral_match` | `{}` | analyze_c_nmr/analyze_h_nmr | ✅ M2 |
| `check_substructure` | `{pattern}` | check_substructure | ✅ M2 |
| `functional_group_scan` | `{}` | check_substructure(兜底) | ✅ M2 |

## 文件

- `nmr_matching_runtime.py` — 确定性工具运行时（handlers 派发表；4 工具全含）。
- `named_smarts.py` — 从 operators.py 提取的 NAMED_SMARTS 表 + 关键词→模式解析（check_substructure/converter 用）。
- `verl/tools/nmr_matching_tools.py` — verl 工具包装 `NmrMatchingTool` + `NmrMatchingSessionManager` 单例。
  `create()` 从 raw_prompt 的 user JSON 取候选+峰；`execute()` 派发到 runtime。加载预测缓存。
- `tool_config_matching.yaml` — 工具 schema（M0 只挂 count_atoms；M2 起需补 3 个 entry 用于 RL）。
- `prompt.py` — system prompt（Hermes 契约：`<think>`/`<tool_call>`/`<tool_response>`/`<answer>`）。
- `precompute_matching_cache.py` — 全量候选 NMR 正向预测缓存（唯一触网步骤）。
- `cot_to_trajectory.py` — CoT→Hermes 多轮工具轨迹转换器（重跑工具生成确定性观测）。
- `standalone_infer_matching.py` — 独立多轮推理，验证 model→tool_call→observation→answer 全链路。
- `test_tools_m0.py` / `test_tools_m2.py` — 单测（纯 CPU，无网络）。
- `prediction_cache.json` — 全量预测缓存（5352 条，100% ok）。
- `data/tool_sft_v4.jsonl` — 转换出的工具 SFT 冷启轨迹（2134 条）。

## M0 验证状态（已完成）

```bash
cd /fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL
python recipe/nmr_matching_agent/test_tools_m0.py   # 全部通过
```

已验证（无需 GPU）：
- ✅ runtime count_atoms C/H 精确计数（对 insert_ch2 负、矛盾例、正样本）。
  **发现**：v4 CoT 首例 Gemini 称"至少 23 个碳"，RDKit 实际 22——确定性工具修正了 CoT 事实错误。
- ✅ `NmrMatchingTool` create/execute/release 生命周期从 raw_prompt user JSON 读候选+峰。
- ✅ standalone 脚本离线部分：alpaca input 解析、action 解析（tool+answer）、工具派发。
- ✅ ckpt-200 tokenizer 的 chat_template 正确渲染多轮工具对话（Qwen3 Hermes 格式）。

**未验证（需大显存 + 应放到 M3 之后）**：真实模型 in-the-loop 解码。原始 ckpt-200 从没见过工具语法，
现在跑只会确认它不出 `<tool_call>`；真正的链路验证在工具 SFT 冷启（M3）后用合并权重跑：

```bash
python recipe/nmr_matching_agent/standalone_infer_matching.py \
  --model_path <tool_sft_merged_ckpt> \
  --from_test /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_test_1000.json \
  --n 2 --output m0_infer.json
```

## M1 全量预测缓存（已完成）

```bash
# dry-run 看覆盖率；--out 触网补齐缺失
python recipe/nmr_matching_agent/precompute_matching_cache.py \
  --parquet recipe/nmr_matching/data/{rl_train_expanded,rl_train,rl_val_1000}.parquet \
  --alpaca /fs_mol/cxq/.../nmr_matching_test_1000.json \
  --cot /fs_mol/cxq/SpecMaster_jp/cot_output/cot_traces_v{4,6}.jsonl \
  --cache recipe/nmr_matching/prediction_cache.json \
  --out recipe/nmr_matching_agent/prediction_cache.json
```
结果：全量候选并集 **5352 条 100% 覆盖、全部 ok**；spectral_match 有区分度 **pos 0.606 > neg 0.430**（v4）。

## M2 全工具 + CoT 转换器（已完成）

```bash
python recipe/nmr_matching_agent/test_tools_m2.py   # 全部通过
python recipe/nmr_matching_agent/cot_to_trajectory.py \
  --cot /fs_mol/cxq/SpecMaster_jp/cot_output/cot_traces_v4.jsonl \
  --pred_cache recipe/nmr_matching_agent/prediction_cache.json \
  --out recipe/nmr_matching_agent/data/tool_sft_v4.jsonl
```
结果：2134 条结构化 CoT 全部转成工具轨迹；**矛盾率仅 3.0%**（远低于 30% 阈值 → CoT 足够可靠，
确定性工具只修正了少量 Gemini 事实错误，如"至少23碳"实为22）。观测均为确定性 JSON，训练/推理一致。

## M3 工具 SFT 冷启（数据就绪，待训练）

工具已全部补齐到 `tool_config_matching.yaml`（5 entry，共享 cache，verl loader 验证通过）。

数据转换链（已验证 LLaMA-Factory 全量 2092 条零丢弃）：
```bash
# 1) CoT -> Hermes 工具轨迹（已产出 data/tool_sft_v4.jsonl, 2134 条）
# 2) -> LLaMA-Factory sharegpt tool 格式
python recipe/nmr_matching_agent/trajectory_to_llamafactory.py \
  --in recipe/nmr_matching_agent/data/tool_sft_v4.jsonl \
  --out /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory/data/nmr_matching_tool_sft_v4.json \
  --val_out .../data/nmr_matching_tool_sft_v4_val.json --val_frac 0.02
```
数据集已注册进 `LLaMA-Factory/data/dataset_info.json`（`nmr_matching_tool_sft_v4` + `_val`），
训练配置 `LLaMA-Factory/examples/train_full/qwen3_32b_nmr_matching_tool_full.yaml`（从 ckpt-200 冷启）。

> 注：system prompt 的工具文档（`prompt.TOOL_DOC_FULL`）已覆盖全部 4 工具，与轨迹实际调用一致；
> `standalone_infer_matching.py` 的 TOOL_SCHEMAS 也已补齐 4 工具、默认加载全量缓存。

**训练命令（需 ~8×A100，本机跑不动 32B 全参 z3）**：
```bash
cd /fs_mol/cxq/Xmaster_AgentRL/repair_sft/LLaMA-Factory
conda activate /fs_mol/cxq/conda_env/JPsft   # 或 source activate
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1   # 非 A100/P2P 机器需要；A100 可省
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen3_32b_nmr_matching_tool_full.yaml
```
关键点：qwen3 模板把 `<think>` 嵌进 function_call turn（thought_words 分隔符须精确
`<think>\n...\n</think>\n\n`，转换器已处理）；观测走 `observation` turn 渲染成 `<tool_response>`。
产出 `saves/qwen3-32b-nmr-matching-tool/full-exp1`。训完后再跑 M0 的 standalone_infer 验证工具调用率。

## 后续里程碑

- **M4** GRPO + 评测（超 SFT 基线 acc 0.729，提升难类）。
