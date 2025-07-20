from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import yaml


# define activation functions
def sigm(x) -> torch.Tensor: 
    return 1 / (1 + torch.exp(-x))

def hardsigm(x) -> torch.Tensor:
    return x.clamp(min=-1).clamp(max=1)

def tanh(x) -> torch.Tensor:
    return torch.tanh(x)

def logexp(x) -> torch.Tensor:
    return torch.log(1 + torch.exp(x))

def softrelu(x) -> torch.Tensor:
    gamma = 0.1
    beta = 1
    theta = 3
    return gamma * torch.log(1 + torch.exp(beta * (x - theta)))

# Create a map to resolve function names in YAML to actual functions
activation_map = {
    "sigm": sigm,
    "hardsigm": hardsigm,
    "tanh": tanh,
    "logexp": logexp,
    "softrelu": softrelu,
}

@dataclass
class Config:
    n_samples: int
    batch_size: int
    dt: float
    t: int
    tau_neu: float
    size_tab: List[int]
    lr_pf: List[float]
    lr_ip: List[float]
    lr_pi: List[float]
    lr_pb: List[float]
    ga: float
    gb: float
    gd: float
    glk: float
    gsom: float
    noise: float
    tau_weights: float
    rho: Callable[[torch.Tensor], torch.Tensor]
    initw: float
    sample_range: Tuple[int, int]

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        cfg["rho"] = activation_map[cfg["rho"]]  # Resolve activation function
        return Config(**cfg)

    def __str__(self):
        """Return a string representation of the Config object."""
        return f"Config: n_samples={self.n_samples}, batch_size={self.batch_size}, dt={self.dt}, t={self.t}, " \
               f"tau_neu={self.tau_neu}, size_tab={self.size_tab}, lr_pf={self.lr_pf}, lr_ip={self.lr_ip}, " \
               f"lr_pi={self.lr_pi}, lr_pb={self.lr_pb}, ga={self.ga}, gb={self.gb}, gd={self.gd}, glk={self.glk}, " \
               f"gsom={self.gsom}, noise={self.noise}, tau_weights={self.tau_weights}, rho={self.rho.__name__}, " \
               f"initw={self.initw}, sample_range={self.sample_range})"
