import torch.optim as optim
from tqdm import tqdm
import numpy as np

from netClasses import *


def create_dataset(
    n_samples, batch_size, input_size, r1, r2, device=torch.device("cpu")
):
    data = (
        torch.FloatTensor(n_samples, batch_size, input_size).uniform_(r1, r2).to(device)
    )
    return data


def evalrun(net, data, targets, config):
    va_topdown_hist = []
    va_cancelation_hist = []
    vb_input_hist = []
    u_p_hist = []
    target_hist = []
    data_trace_hist = data[0][0].unsqueeze(0).clone()
    data_trace = data[0].clone()
    print(config.t)
    for n in range(len(data)):
        print(f"evalrun, sample {n + 1}")
        batch = data[n].to(config.device)
        target = targets[n] if targets else None

        for k in range(config.t):
            # Update data_trace
            data_trace += (config.dt / config.tau_neu) * (-data_trace + batch)
            # Step the network
            den_input = net.stepper(data_trace, target=target)
            va_topdown, va_cancelation, vb_input = den_input
            # check if anything is nan
            if (
                torch.isnan(va_topdown[0]).any()
                or torch.isnan(va_cancelation[0]).any()
                or torch.isnan(vb_input[0]).any()
            ):
                print("nan found")
                print("va_topdown: ", va_topdown)
                print("va_cancelation: ", va_cancelation)
                print("vb_input: ", vb_input)
                print("data_trace: ", data_trace)
                print("batch: ", batch)
                print("target: ", target)
                raise ValueError("nan found")

            # Update data trace history
            data_trace_hist = torch.cat(
                (data_trace_hist, data_trace[0].unsqueeze(0).cpu()), dim=0
            )
            if target is not None:
                target_hist.append(target[0].cpu().numpy())
            else:
                target_hist.append(np.full((net.topology[-1],), np.nan))

            va_topdown_hist = net.updateHist(va_topdown_hist, va_topdown)
            va_cancelation_hist = net.updateHist(va_cancelation_hist, va_cancelation)
            vb_input_hist = net.updateHist(vb_input_hist, vb_input)
            u_p_hist = net.updateHist(u_p_hist, net.u_p)

    return (
        data_trace_hist.numpy(),
        [va_topdown_hist, va_cancelation_hist, vb_input_hist],
        np.array(target_hist),
        u_p_hist,
    )


def self_pred_training(net, train_data, config):
    net.train()
    try:
        wpf_hist = []
        wpb_hist = []
        wpi_hist = []
        wip_hist = []
        va_topdown_hist = []
        va_cancelation_hist = []
        u_p_hist = []
        u_i_hist = []
        data_trace = train_data[0].clone()
        batch = train_data[0].clone()
        data_trace = data_trace.to(config.device)
        batch = batch.to(config.device)
        print(data_trace.device, batch.device)
        for n in tqdm(range(train_data.shape[0])):
            # put data[n] on device
            batch.copy_(train_data[n])
            for k in range(config.t):
                # low-pass filter the data
                data_trace += (config.dt / config.tau_neu) * (-data_trace + batch)
                va = net.stepper(data_trace)
                # Track apical potential, neurons and synapses
                if k == 0 and n % 20 == 0:
                    va_topdown, va_cancelation, vb_input = va
                    # Update the history lists
                    va_topdown_hist = net.updateHist(va_topdown_hist, va_topdown)
                    va_cancelation_hist = net.updateHist(
                        va_cancelation_hist, va_cancelation
                    )
                    wpf_hist = net.updateHist(wpf_hist, net.wpf, param=True)
                    wpb_hist = net.updateHist(wpb_hist, net.wpb, param=True)
                    wpi_hist = net.updateHist(wpi_hist, net.wpi, param=True)
                    wip_hist = net.updateHist(wip_hist, net.wip, param=True)
                    u_p_hist = net.updateHist(u_p_hist, net.u_p)
                    u_i_hist = net.updateHist(u_i_hist, net.u_i)

                # Update the pyramidal-to-interneuron weights (NOT the pyramidal-to-pyramidal weights !)
                net.updateWeights(batch)
            if n % 100 == 0:
                apical_dist = ((va[0][0] + va[1][0]) ** 2).cpu().numpy().sum(1).mean(0)
                pyr_int_dist = (
                    ((net.u_p[-1] - net.u_i[0]) ** 2).cpu().numpy().sum(1).mean(0)
                )
                print("apical_dist: ", apical_dist, "pyr_int_dist: ", pyr_int_dist)
    except KeyboardInterrupt:
        pass

    return (
        [va_topdown_hist, va_cancelation_hist],
        [wpf_hist, wpb_hist, wpi_hist, wip_hist],
        [u_p_hist, u_i_hist],
        n + 1,
    )


def target_training(steps, net, data, rand_target, t, dt, tau_neu, device):
    net.to(device)
    net.train()
    try:
        # initialize hist lists
        apical_dist_hist = [[] for _ in range(net.depth - 1)]
        pyr_int_dist_hist = [[] for _ in range(net.depth - 1)]
        val_error_hist = []
        # prepare data
        data_trace = data.clone()
        data_trace = data_trace.to(device)

        # training loop
        for n in tqdm(range(steps)):
            # introduce target after 1000 steps
            if n == 0:
                target = None
            else:
                target = rand_target
            for k in range(t):
                # low-pass filter the data
                data_trace += (dt / tau_neu) * (-data_trace + data)
                va = net.stepper(data_trace, target)
                va_topdown, va_cancelation, vb_input = va
                # Update the pyramidal-to-interneuron weights (NOT the pyramidal-to-pyramidal weights !)
                net.updateWeights(data)
            for k in range(net.depth - 1):
                apical_dist_hist[k].append(
                    ((va_topdown[k] + va_cancelation[k]) ** 2)
                    .cpu()
                    .numpy()
                    .sum(1)
                    .mean(0)
                )
                pyr_int_dist_hist[k].append(
                    ((net.s[k + 1] - net.i[k]) ** 2).cpu().numpy().sum(1).mean(0)
                )
            if target is not None:
                val_error_hist.append(
                    ((net.s[-1] - target) ** 2).cpu().numpy().sum(1).mean(0)
                )
            else:
                val_error_hist.append(None)

    except KeyboardInterrupt:
        pass

    return (
        apical_dist_hist,
        pyr_int_dist_hist,
        val_error_hist,
    )


def nonlinear_regression_training(steps, net, data, target, t, dt, tau_neu, device):
    pass
