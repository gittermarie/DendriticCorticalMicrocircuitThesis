import torch


# define activation functions
def sigm(x):
    return 1 / (1 + torch.exp(-x))


def hardsigm(x):
    return x.clamp(min=-1).clamp(max=1)


def tanh(x):
    return torch.tanh(x)


def logexp(x):
    return torch.log(1 + torch.exp(x))


def softrelu(x):
    gamma = 0.1
    beta = 1
    theta = 3
    return gamma * torch.log(1 + torch.exp(beta * (x - theta)))


class Config:
    def __init__(
        self,
        name,
        n_samples,
        batch_size,
        dt,
        t,
        tau_neu,
        size_tab,
        lr_pf,
        lr_ip,
        lr_pi,
        lr_pb,
        ga,
        gb,
        gd,
        glk,
        gsom,
        noise,
        tau_weights,
        rho,
        initw,
        device,
        sample_range,
    ):
        self.name = name
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.dt = dt
        self.t = t
        self.tau_neu = tau_neu
        self.size_tab = size_tab
        self.lr_pf = lr_pf
        self.lr_ip = lr_ip
        self.lr_pi = lr_pi
        self.lr_pb = lr_pb
        self.ga = ga
        self.gb = gb
        self.gd = gd
        self.glk = glk
        self.gsom = gsom
        self.noise = noise
        self.tau_weights = tau_weights
        self.rho = rho
        self.initw = initw
        self.device = device
        self.sample_range = sample_range

    @staticmethod
    def fig_s1():
        return Config(
            name="fig_s1",
            n_samples=10000,
            batch_size=1,
            dt=0.1,
            t=1000,
            tau_neu=3,
            size_tab=[30, 20, 10],
            lr_pf=[0, 0],
            lr_ip=[0.0011875],
            lr_pi=[0.0005],
            lr_pb=[0],
            ga=0.8,
            gb=1.0,
            gd=1.0,
            glk=0.1,
            gsom=0.8,
            noise=0.1,
            tau_weights=30,
            rho=sigm,  # logexp,
            initw=1,
            device=torch.device("cpu"),
            sample_range=(0, 1),
        )

    @staticmethod
    def fig_1():
        return Config(
            name="fig_1",
            n_samples=1000,
            batch_size=1,
            dt=0.1,
            t=500,
            tau_neu=2,
            size_tab=[30, 50, 10],
            lr_pf=[0.001, 0.0005],
            lr_ip=[0.001],
            lr_pi=[0.005],
            lr_pb=[0],
            ga=0.7,
            gb=1,
            gd=1,
            glk=0.1,
            gsom=0.7,
            noise=0.2,
            tau_weights=20,
            rho=softrelu,
            initw=0.5,
            device=torch.device("cpu"),
            sample_range=(0, 1),
        )

    @staticmethod
    def fig_2():
        return Config(
            name="fig_2",
            n_samples=75,
            batch_size=75,
            dt=0.1,
            t=750,
            tau_neu=4,
            size_tab=[40, 30, 20],
            lr_pf=[0.0015, 0.0007],
            lr_ip=[0.0015],
            lr_pi=[0.0007],
            lr_pb=[0],
            ga=0.9,
            gb=1,
            gd=1,
            glk=0.1,
            gsom=0.9,
            noise=0.15,
            tau_weights=25,
            rho=tanh,
            initw=0.8,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            sample_range=(0, 1),
        )

    def __repr__(self):
        return f"name={self.name}\nsamples={self.n_samples}\nbatch_size={self.batch_size}\ndt={self.dt}\nt={self.t}\ntau_neu={self.tau_neu}\nsize_tab={self.size_tab}\nlr_pf={self.lr_pf}\nlr_ip={self.lr_ip}\nlr_pi={self.lr_pi}\nlr_pb={self.lr_pb}\nga={self.ga}\ngb={self.gb}\ngd={self.gd}\nglk={self.glk}\ngsom={self.gsom}\nnoise={self.noise}\ntau_weights={self.tau_weights}\nrho={self.rho.__name__}\ninitw={self.initw}\ndevice={self.device}\nsample_range={self.sample_range}"
