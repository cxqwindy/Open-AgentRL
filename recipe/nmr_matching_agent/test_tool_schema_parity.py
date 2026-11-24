from pathlib import Path
import yaml
from recipe.nmr_matching_agent.standalone_infer_matching import TOOL_SCHEMAS
from recipe.nmr_matching_agent.trajectory_to_llamafactory import TOOL_SCHEMAS as LF_SCHEMAS
from recipe.nmr_matching_agent.reward_matching_agent import TOOL_NAMES
from recipe.nmr_matching_agent.prompt import TOOL_DOC_FULL

EXPECTED={"count_atoms","spectral_match","check_substructure","functional_group_scan","peak_assignment"}

def test_five_tool_contract_parity():
    standalone=set(TOOL_SCHEMAS)
    lf={x["name"] for x in LF_SCHEMAS}
    cfg=yaml.safe_load(Path("recipe/nmr_matching_agent/tool_config_matching.yaml").read_text())
    yml={x["tool_schema"]["function"]["name"] for x in cfg["tools"]}
    assert standalone==lf==yml==TOOL_NAMES==EXPECTED
    for name in EXPECTED: assert name in TOOL_DOC_FULL
    peak=TOOL_SCHEMAS["peak_assignment"]["function"]["parameters"]
    assert set(peak["properties"])=={"nucleus"}
