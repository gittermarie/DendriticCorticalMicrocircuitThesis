import torch.optim as optim
from tqdm import tqdm
import numpy as np

from netClasses import *


def create_dataset(n_samples, batch_size, input_size, mu, sigma, device=torch.device("cpu")):
    data = (
        2 * sigma * torch.rand((n_samples, batch_size, input_size), device=device)
        - sigma
        + mu
    )
    return data


def validate_self_pred(net, data, t, dt, tau_neu, device):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    net.eval()
    val_error = 0.0
    with torch.inference_mode():
        # validate
        data_trace = data[0].clone()
        for d in tqdm(data):
            for k in range(t):
                data_trace += (dt / tau_neu) * (-data_trace + d)
                va = net.stepper(data_trace, track_va=True)
                va_topdown, va_cancelation = va
            for k in range(1, net.net_depth):
                val_error += (
                    ((va_topdown[k - 1] + va_cancelation[k - 1]) ** 2)
                    .cpu()
                    .numpy()
                    .mean(1)[0]
                )

    return val_error / data.shape[0]


def validate_nonlinear_regression(net, data, teacher_net, t, dt, tau_neu, device):
    net.eval()
    val_error = 0.0
    val_loss = 0.0

    with torch.inference_mode():
        data_trace = data[0].clone()
        for d in tqdm(data):
            target = teacher_net(data)
            for k in range(t):
                data_trace += (dt / tau_neu) * (-data_trace + d)
                va = net.stepper(data_trace, target=target)
                va_topdown, va_cancelation = va
            for k in range(1, net.net_depth):
                val_error += (
                    ((va_topdown[k - 1] - va_cancelation[k - 1]) ** 2)
                    .cpu()
                    .numpy()
                    .mean(1)[0]
                )
            val_loss += ((net.s[-1] - target) ** 2).cpu().numpy().mean(1)[0]

    return val_error / data.shape[0], val_loss / data.shape[0]


def evalrun(net, data, targets, batch_size, t, dt, tau_neu, device):
    va_topdown_hist = []
    va_cancelation_hist = []
    s_hist = []
    target_hist = []
    data_trace = data[0].clone()
    # move data trace onto device
    data_trace = data_trace.to(device)
    data_trace_hist = data[0].unsqueeze(2).mean(0)
    batch = data[0].clone()
    batch = batch.to(device)
    print(data_trace.device, batch.device)
    for n in range(len(data)):
        print("evalrun, sample {}".format(1 + n))
        # fill batch array on device with values from data(n)
        batch.copy_(data[n]) 
        for k in range(t):
            # low-pass filter the data
            data_trace += (dt / tau_neu) * (-data_trace + batch)
            # Step the neural network
            va = net.stepper(data_trace, target=targets[n])
            # Track apical potential, neurons and synapses
            va_topdown, va_cancelation = va
            data_trace_hist = torch.cat(
                (data_trace_hist, data_trace[0].unsqueeze(1).cpu()), dim=1
            )
            if targets[n] is None:
                # fill numpy array of shape (BATCH_SIZE, SIZE_TAB_TEACHER, 1) with numpy.nan
                target_hist.append(
                    np.full((net.net_topology[-1], 1), np.nan)
                )
            else:
                target_hist.append(targets[n][0].clone().unsqueeze(1).cpu().numpy())
            va_topdown_hist = net.updateHist(va_topdown_hist, va_topdown)
            va_cancelation_hist = net.updateHist(va_cancelation_hist, va_cancelation)
            s_hist = net.updateHist(s_hist, net.s)
    return (
        data_trace_hist,
        va_topdown_hist,
        va_cancelation_hist,
        np.array(target_hist),
        s_hist,
    )


def self_pred_training(net, data, t, dt, tau_neu, device):
    net.to(device)
    net.train()
    try:
        wpf_hist = []
        wpb_hist = []
        wpi_hist = []
        wip_hist = []
        va_topdown_hist = []
        va_cancelation_hist = []
        data_trace = data[0].clone()
        batch = data[0].clone()
        data_trace = data_trace.to(device)
        batch = batch.to(device)
        print(data_trace.device, batch.device)
        for n in tqdm(range(data.shape[0])):
            # put data[n] on device
            batch.copy_(data[n])
            for k in range(t):
                # low-pass filter the data
                data_trace += (dt / tau_neu) * (-data_trace + batch)
                va = net.stepper(data_trace)
                # Track apical potential, neurons and synapses
                if k == 0 and n % 20 == 0:
                    va_topdown, va_cancelation = va
                    # Update the tabs with the current values
                    va_topdown_hist = net.updateHist(va_topdown_hist, va_topdown)
                    va_cancelation_hist = net.updateHist(
                        va_cancelation_hist, va_cancelation
                    )
                    wpf_hist = net.updateHist(wpf_hist, net.wpf, param=True)
                    wpb_hist = net.updateHist(wpb_hist, net.wpb, param=True)
                    wpi_hist = net.updateHist(wpi_hist, net.wpi, param=True)
                    wip_hist = net.updateHist(wip_hist, net.wip, param=True)

                # Update the pyramidal-to-interneuron weights (NOT the pyramidal-to-pyramidal weights !)
                net.updateWeights(batch)
    except KeyboardInterrupt:
        pass

    net.save_weights(r"weights_{}_{}.pt".format(n + 1, net.net_topology))

    return (
        [va_topdown_hist, va_cancelation_hist],
        wpf_hist,
        wpb_hist,
        wpi_hist,
        wip_hist,
        n + 1,
    )


def target_training(net, data, target_net, t, dt, tau_neu):
    try:
        data_trace = data[0].clone()
        for n in tqdm(range(data.shape[0])):
            if n == 0:
                target = None
            else:
                target = target_net(data[n])

            for k in range(t):

                # low-pass filter the data
                data_trace += (dt / tau_neu) * (-data_trace + data[n])

                # Step the neural network
                va = net.stepper(data_trace, target=target)

                # Update the pyramidal-to-interneuron weights (INCLUDING the pyramidal-to-pyramidal weights !)
                net.updateWeights(data[n], target=target)

    except KeyboardInterrupt:
        pass
