import yaml


class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                if k == "ssm_cfg":
                    setattr(self, k, None)
                else:
                    setattr(self, k, Config(v))
            else:
                # print(k, v)
                setattr(self, k, v)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return Config(config)
