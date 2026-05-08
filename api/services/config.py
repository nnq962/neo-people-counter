import yaml
import os

CONFIG_PATH = "configs/default.yaml"

class FlowList(list):
    """Lớp hỗ trợ in mảng thành chuỗi inline [x, y] trong YAML."""
    pass

def flow_list_rep(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowList, flow_list_rep)

def get_config_data() -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError("Configuration file not found.")
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def deep_update(d, u):
    import collections.abc
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def save_config_data(config_dict: dict) -> None:
    # Ép kiểu điểm (points) thành mảng inline [x, y] thay vì list nhiều dòng
    if "zone" in config_dict and config_dict["zone"] is not None:
        if "points" in config_dict["zone"]:
            config_dict["zone"]["points"] = [FlowList(pt) for pt in config_dict["zone"]["points"]]

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
