import torch.optim as optim
import numpy as np

from netClasses_eff import *
from plotFunctions import *
from training_and_eval import *

# input batch size for training (default: 0)
BATCH_SIZE = 1

# time discretization (default: 0.1)
DT = 0.1

# number of time steps per sample (default: 1000)
T = 1000

# time constant of the input patterns (default: 3)
TAU_NEU = 3

# freeze the dynamics of the feedback weights (default: False)
FREEZE_FEEDBACK = False

# use biases (default: False)
BIAS = False

# architecture of the teacher net (default: [30, 20, 10])
SIZE_TAB_TEACHER = [30, 20, 10]

# architecture of the net (default: [30, 20, 10])
K_TAB = [2, 10]


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


def fig_s1(net, device):
    net.to(device)
    with torch.no_grad():
        print("---before learning self-prediction---")
        eval_data = [
            2 * torch.rand(BATCH_SIZE, net.net_topology[0], device=device) - 1
            for _ in range(3)
        ]
        targets = [None for _ in range(3)]
        data_trace_hist, va_topdown_hist, va_cancelation_hist, target_hist, s_hist = (
            evalrun(
                net,
                eval_data,
                targets,
                BATCH_SIZE,
                T,
                DT,
                TAU_NEU,
                device,
            )
        )
        plot_apical_trace(
            "pre-learning_selfpred_eval_apicaltrace",
            5,
            net.net_depth + 1,
            data_trace_hist,
            va_topdown_hist,
            va_cancelation_hist,
            target_hist,
            s_hist,
        )
        print("---learning self-prediction---")
        data = create_dataset(5000, BATCH_SIZE, net.net_topology[0], 0, 1, device)
        va, wpf_hist, wpb_hist, wpi_hist, wip_hist, n = self_pred_training(
            net, data, T, DT, TAU_NEU, device
        )
        plot_synapse_distance(
            r"learning_selfpred({}eps)synapsedistance".format(n),
            net.net_depth + 1,
            wpf_hist,
            wpb_hist,
            wpi_hist,
            wip_hist,
        )
        plot_synapse_trace(
            r"learning_selfpred({}eps)_synapsetrace".format(n),
            net.net_depth + 1,
            wpf_hist,
            wpb_hist,
            wpi_hist,
            wip_hist,
        )
        plot_apical_distance(
            r"learning_selfpred({}eps)_apicaldistance".format(n),
            net.net_depth + 1,
            va[0],
            va[1],
        )
        print("---after learning self-prediction---")
        data_trace_hist, va_topdown_hist, va_cancelation_hist, target_hist, s_hist = (
            evalrun(
                net,
                eval_data,
                targets,
                BATCH_SIZE,
                T,
                DT,
                TAU_NEU,
                device,
            )
        )
        plot_apical_trace(
            "post-learning_selfpred_eval_apicaltrace",
            5,
            net.net_depth + 1,
            data_trace_hist,
            va_topdown_hist,
            va_cancelation_hist,
            target_hist,
            s_hist,
        )
        plt.show()


def fig_1(net, device, train_from_scratch=False):
    net.lr_pf = [0.0011875, 0.0005]
    with torch.no_grad():
        if train_from_scratch:
            data = create_dataset(10000, BATCH_SIZE, net.net_topology[0], 0, 1, device)
            self_pred_training(net, data, T, DT, TAU_NEU, device)
        else:
            net.load_weights(
                r"weights/2024-12-05/weights_10000_{}.pt".format(net.net_topology)
            )
            net.to(device)
            net.train()

        global teacherNet
        # Build the teacher net
        teacherNet = teacherNet(SIZE_TAB_TEACHER, K_TAB)

        # pre-training evaluation
        data = create_dataset(1, BATCH_SIZE, net.net_topology[0], 0, 1, device)
        target = teacherNet.forward(data)
        data_trace_hist, va_topdown_hist, va_cancelation_hist, target_hist, s_hist = (
            evalrun(
                net,
                [data, data],
                [None, target],
                BATCH_SIZE,
                T,
                DT,
                TAU_NEU,
                device,
            )
        )
        plot_apical_trace(
            "pre-training_target_eval_apicaltrace",
            5,
            net.net_depth + 1,
            data_trace_hist,
            va_topdown_hist,
            va_cancelation_hist,
            target_hist,
            s_hist,
        )
        # target training
        n = 15
        s, i = net.initHidden(device=device)
        target_training(net, data, target, s, i, T, DT, TAU_NEU)
        # post-training evaluation
        data_trace_hist, va_topdown_hist, va_cancelation_hist, target_hist, s_hist = (
            evalrun(
                net,
                [data, data],
                [None, target],
                BATCH_SIZE,
                T,
                DT,
                TAU_NEU,
                device,
            )
        )
        plot_apical_trace(
            r"post-training_target_eval({}eps)_apicaltrace".format(n),
            5,
            net.net_depth + 1,
            data_trace_hist,
            va_topdown_hist,
            va_cancelation_hist,
            target_hist,
            s_hist,
        )
        plt.show()


def fig_2(net, device, train_from_scratch=False):
    net.to(device)
    net.train()

    with torch.no_grad():
        # train for self-predicting state
        if train_from_scratch:
            net.load_weights(
                r"weights/2024-12-04/weights_10000_{}.pt".format(net.net_topology)
            )
        else:
            self_pred_training(net, 10000, BATCH_SIZE, T, DT, TAU_NEU, device)
        net.to(device)
        # train non-linear regression task
        global teacherNet
        teacherNet = teacherNet(SIZE_TAB_TEACHER, K_TAB)
        teacherNet.to(device)
        s, i = net.initHidden(device=device)
        try:
            for n in range(1000):
                data = torch.rand(BATCH_SIZE, net.net_topology[0], device=device)
                target = teacherNet.forward(data)
                s, i = target_training(15, net, data, target, s, i, T, DT, TAU_NEU)
        except KeyboardInterrupt:
            pass

        net.save_weights(
            r"weights/2024-12-05/weights_target_{}_{}.pt".format(
                n + 1, net.net_topology
            )
        )

        plt.show()


if __name__ == "__main__":
    # Define the device
    print(torch.cuda.is_available())
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)

    # Define the networks
    net_1 = dendriticNet(
        T,
        DT,
        BATCH_SIZE,
        size_tab=[30, 20, 10],
        lr_pf=[0, 0],
        lr_ip=[0.0011875],
        lr_pi=[0.0005],
        lr_pb=[0],
        ga=0.8,
        gb=1,
        gd=1,
        glk=0.1,
        gsom=0.8,
        noise=0.1,
        tau_weights=30,
        rho=logexp,
        initw=1,
    )
    net_2 = dendriticNet(
        T,
        DT,
        BATCH_SIZE,
        size_tab=[30, 50, 10],
        lr_pf=[0.0011875, 0.0005],
        lr_ip=[0.0011875],
        lr_pi=[0.0059375],
        lr_pb=[0],
        ga=0.8,
        gb=1,
        gd=1,
        glk=0.1,
        gsom=0.8,
        noise=0.3,
        tau_weights=30,
        rho=softrelu,
        initw=0.1,
    )

    fig_s1(net_1, device)
    # fig_1(net_1, device)
    # fig_2(net_2, device)
