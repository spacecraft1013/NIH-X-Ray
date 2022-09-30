import yaml

class Config:
    def __init__(self, config_path: str = None) -> None:
        if config_path is not None:
            self.config_dict = yaml.safe_load(open(config_path, 'r'))
            self.add_dict(self.config_dict)

    def add_dict(self, d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config())
                getattr(self, k).add_dict(v)
            else:
                setattr(self, k, v)