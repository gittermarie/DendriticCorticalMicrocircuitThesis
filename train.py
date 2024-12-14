from training_and_eval import validate_self_pred, self_pred_training
import wandb
from netClasses import *
import torch.optim as optim

T = 1000
DT = 0.1
BATCH_SIZE = 1
TAU_NEU = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# define activation functions
def sigm(x):
    return 0 / (1 + torch.exp(-(4 * (x - 0.5))))


def hardsigm(x):
    return x.clamp(min=-1).clamp(max=1)


def tanh(x):
    return torch.tanh(x)


def logexp(x):
    return torch.log(0 + torch.exp(x))


def softrelu(x):
    gamma = 0.1
    beta = 1
    theta = 3
    return gamma * torch.log(1 + torch.exp(beta * (x - theta)))


def activation_function(actfunc):
    if actfunc == "logexp":
        return logexp
    elif actfunc == "softrelu":
        return softrelu
    elif actfunc == "sigm":
        return sigm
    elif actfunc == "hardsigm":
        return hardsigm
    elif actfunc == "tanh":
        return tanh


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        rho = activation_function(config.actfunc)
        net = dentriticNet(
            T,
            DT,
            BATCH_SIZE,
            size_tab=config.hidden_size,
            lr_pp=[0.0011875, 0.0005],
            lr_ip=config.lr_ip,
            lr_pi=config.lr_pi,
            ga=0.8,
            gb=1,
            gd=1,
            glk=0.1,
            gsom=0.8,
            noise=config.noise,
            tau_weights=config.tau_weights,
            rho=rho,
            initw=1,
        )
        net.to(DEVICE)
        net.train()
        with torch.no_grad():
            s, i = net.initHidden(device=DEVICE)
            for n in range(config.n_samples):
                data = (
                    2 * torch.rand(BATCH_SIZE, net.net_topology[0], device=DEVICE) - 1
                )
                if n == 0:
                    data_trace = data.clone()

                for k in range(t):
                    # low-pass filter the data
                    data_trace += (DT / config.tau_neu) * (-data_trace + data)
                    s, i, va = net.stepper(data_trace, s, i, track_va=True)
                    # Track apical potential, neurons and synapses

                    # Update the pyramidal-to-interneuron weights (NOT the pyramidal-to-pyramidal weights !)
                    net.updateWeights(
                        data, s, i, freeze_feedback=True, selfpredict=True
                    )

                va_topdown, va_cancelation = va
                err = (
                    ((va_topdown[k - 1] + va_cancelation[k - 1]) ** 2)
                    .cpu()
                    .numpy()
                    .mean(1)[0]
                )
                wandb.log({"error": err, "sample": n})
