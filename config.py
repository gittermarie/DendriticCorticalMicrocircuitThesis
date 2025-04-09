from pathlib import Path

import torch
import yaml


# define activation functions
def sigm(x): return 1 / (1 + torch.exp(-x))
def hardsigm(x): return x.clamp(min=-1).clamp(max=1)
def tanh(x): return torch.tanh(x)
def logexp(x): return torch.log(1 + torch.exp(x))
def softrelu(x): return 0.1 * torch.log(1 + torch.exp(1 * (x - 3)))

activation_map = {
    "sigm": sigm,
    "hardsigm": hardsigm,
    "tanh": tanh,
    "logexp": logexp,
    "softrelu": softrelu,
}

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def from_yaml(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        cfg["rho"] = activation_map[cfg["rho"]]
        cfg["device"] = torch.device(cfg["device"])
        return Config(**cfg)

    def __repr__(self):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.tolist()
        return str(self.__dict__)
        
