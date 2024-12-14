import torch.optim as optim
from tqdm import tqdm
import numpy as np

from netClasses import *


def create_dataset(n_samples, batch_size, input_size, mu, sigma, device):
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
        s, i = net.initHidden()
        # validate
        data_trace = data[0].clone()
        for d in tqdm(data):
            for k in range(t):
                data_trace += (dt / tau_neu) * (-data_trace + d)
                s, i, va = net.stepper(data_trace, s, i, track_va=True)
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
        s, i = net.initHidden()
        data_trace = data[0].clone()
        for d in tqdm(data):
            target = teacher_net(data)
            for k in range(t):
                data_trace += (dt / tau_neu) * (-data_trace + d)
                s, i, va = net.stepper(data_trace, s, i, target=target, track_va=True)
                va_topdown, va_cancelation = va
            for k in range(1, net.net_depth):
                val_error += (
                    ((va_topdown[k - 1] - va_cancelation[k - 1]) ** 2)
                    .cpu()
                    .numpy()
                    .mean(1)[0]
                )
            val_loss += ((s[-1] - target) ** 2).cpu().numpy().mean(1)[0]

    return val_error / data.shape[0], val_loss / data.shape[0]


def evalrun(net, samples, targets, batch_size, t, dt, tau_neu, device):
    s, i = net.initHidden(device=device)
    va_topdown_hist = []
    va_cancelation_hist = []
    s_hist = []
    target_hist = []
    data_trace = samples[0].clone()
    data_trace_hist = samples[0].unsqueeze(2)
    for n, sample in enumerate(samples):
        print("evalrun, sample {}".format(1 + n))
        for k in range(t):
            # low-pass filter the data
            data_trace += (dt / tau_neu) * (-data_trace + sample)
            # Step the neural network
            s, i, va = net.stepper(data_trace, s, i, track_va=True, target=targets[n])
            # Track apical potential, neurons and synapses
            va_topdown, va_cancelation = va

            # Update the tabs with the current values
            data_trace_hist = torch.cat(
                (data_trace_hist, data_trace.unsqueeze(2)), dim=2
            )
            if targets[n] is None:
                # fill numpy array of shape (BATCH_SIZE, SIZE_TAB_TEACHER, 1) with numpy.nan
                target_hist.append(
                    np.full((batch_size, net.net_topology[-1], 1), np.nan)
                )
            else:
                target_hist.append(targets[n].clone().unsqueeze(2).cpu().numpy())
            va_topdown_hist = net.updateHist(va_topdown_hist, va_topdown)
            va_cancelation_hist = net.updateHist(va_cancelation_hist, va_cancelation)
            s_hist = net.updateHist(s_hist, s)
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
        s, i = net.initHidden(device=device)
        wpf_hist = []
        wpb_hist = []
        wpi_hist = []
        wip_hist = []
        va_topdown_hist = []
        va_cancelation_hist = []
        data_trace = data[0].clone()
        for n in tqdm(range(data.shape[0])):
            for k in range(t):
                # low-pass filter the data
                data_trace += (dt / tau_neu) * (-data_trace + data[n])
                s, i, va = net.stepper(data_trace, s, i, track_va=True)
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
                net.updateWeights(data[n], s, i, freeze_feedback=True, selfpredict=True)
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


def target_training(net, data, target_net, s, i, t, dt, tau_neu):
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
                s, i = net.stepper(data_trace, s, i, target=target)

                # Update the pyramidal-to-interneuron weights (INCLUDING the pyramidal-to-pyramidal weights !)
                net.updateWeights(data[n], s, i, freeze_feedback=True)

    except KeyboardInterrupt:
        pass
    return s, i
