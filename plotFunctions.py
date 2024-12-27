import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pickle
import torch
import datetime


def save_graphic(filename, fig):
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    hour = now.strftime("%H")
    # make sure the directory graphics/date_string exists
    os.makedirs(r"graphics/" + date_string, exist_ok=True)
    fig.savefig(
        r"graphics/" + date_string + "/" + filename + "_" + hour + ".png",
        pad_inches=0,
    )


def plot_synapse_distance(fig_title, net_depth, wpf_hist, wpb_hist, wpi_hist, wip_hist):
    fig = plt.figure(figsize=(12, net_depth))
    plt.rcParams.update({"font.size": 12})
    # set figure title
    plt.suptitle(fig_title)
    for i in range(1, net_depth - 1):
        sqrd_frob_norm = (
            torch.norm(wpf_hist[i] - wip_hist[i - 1], p="fro", dim=(0, 1)) ** 2
        )
        # plot squared frobenious matrix norm between wpi and wpf and wip and wpb
        plt.subplot(net_depth - 2, 2, i)
        plt.plot(
            0.1 * np.linspace(0, wpf_hist[i].size(2) - 1, wpf_hist[i].size(2)),
            sqrd_frob_norm.cpu().numpy(),
            color="grey",
            linewidth=3,
            alpha=0.8,
        )
        plt.xlabel(r"$||W^{\rm pp, forward} - W^{\rm ip}||^2$")
        sqrd_frob_norm = (
            torch.norm(wpb_hist[i - 1] - wpi_hist[i - 1], p="fro", dim=(0, 1)) ** 2
        )
        plt.grid()
        plt.subplot(net_depth - 2, 2, i + 1)
        plt.plot(
            0.1 * np.linspace(0, wpf_hist[i].size(2) - 1, wpf_hist[i].size(2)),
            sqrd_frob_norm.cpu().numpy(),
            color="grey",
            linewidth=3,
            alpha=0.8,
        )
        plt.xlabel(r"$||W^{\rm pp, backward} - W^{\rm pi}||^2$")
        plt.grid()

    save_graphic(fig_title, fig)


def plot_synapse_trace(fig_title, net_depth, wpf_hist, wpb_hist, wpi_hist, wip_hist):
    weight_hists = [wpf_hist, wpb_hist, wpi_hist, wip_hist]
    weight_colors = ["red", "blue", "orange", "purple"]
    weight_lables = [
        r"$W^{\rm pp, forward}$",
        r"$W^{\rm pp, backward}$",
        r"$W^{\rm pi}$",
        r"$W^{\rm ip}$",
    ]

    fig = plt.figure(figsize=(12, net_depth))
    plt.rcParams.update({"font.size": 12})
    # set figure title
    plt.suptitle(fig_title)
    # plot input weights
    plt.subplot(net_depth - 1, 4, 1)
    # plot mean across all synapses

    plt.plot(
        0.1 * np.linspace(0, wpf_hist[0].size(2) - 1, wpf_hist[0].size(2)),
        wpf_hist[0].mean(0).mean(0).cpu().numpy(),
        color="red",
        linewidth=3,
        alpha=0.8,
    )

    # plot ten random synapses
    for j in range(10):
        # sample random synapse
        ind_0, ind_1 = np.random.randint(wpf_hist[0].size(0)), np.random.randint(
            wpf_hist[0].size(1)
        )
        plt.plot(
            0.1 * np.linspace(0, wpf_hist[0].size(2) - 1, wpf_hist[0].size(2)),
            wpf_hist[0][ind_0, ind_1, :].cpu().numpy(),
            color="red",
            linewidth=1.5,
            alpha=0.2,
        )

    plt.xlabel("Time (ms)")
    plt.ylabel("Layer 1")
    plt.grid()

    # Loop over layers
    for i in range(1, net_depth - 1):
        for w in range(len(weight_hists)):
            plt.subplot(net_depth - 1, 4, (w + 1) + 4 * i)
            if w == 0:
                plt.ylabel("Layer {}".format(1 + i))
                idx_sub = 0
            else:
                idx_sub = 1

            # plot mean across all synapses
            plt.plot(
                0.1
                * np.linspace(
                    0,
                    weight_hists[w][i - idx_sub].size(2) - 1,
                    weight_hists[w][i - idx_sub].size(2),
                ),
                weight_hists[w][i - idx_sub].mean(0).mean(0).cpu().numpy(),
                color=weight_colors[w],
                linewidth=3,
                alpha=0.8,
            )

            # plot ten random synapses
            for j in range(10):
                # sample random synapse
                ind_0, ind_1 = np.random.randint(
                    weight_hists[w][i - idx_sub].size(0)
                ), np.random.randint(weight_hists[w][i - idx_sub].size(1))
                plt.plot(
                    0.1
                    * np.linspace(
                        0,
                        weight_hists[w][i - idx_sub].size(2) - 1,
                        weight_hists[w][i - idx_sub].size(2),
                    ),
                    weight_hists[w][i - idx_sub][ind_0, ind_1, :].cpu().numpy(),
                    color=weight_colors[w],
                    linewidth=1.5,
                    alpha=0.2,
                )

            plt.xlabel(weight_lables[w])
            plt.grid()

    fig.tight_layout()
    save_graphic(fig_title, fig)


def plot_neuron_trace(fig_title, net_depth, data_trace_hist, s_hist):
    fig = plt.figure(figsize=(5, 2 * net_depth))
    plt.rcParams.update({"font.size": 12})
    # set figure title
    plt.suptitle(fig_title)
    # plot sensory input
    plt.subplot(net_depth, 1, 1)
    plt.plot(
        0.1 * np.linspace(0, data_trace_hist.size(2) - 1, data_trace_hist.size(2)),
        data_trace_hist.mean(1).mean(0).cpu().numpy(),
        color="green",
        linewidth=3,
        alpha=0.8,
    )
    for j in range(10):
        plt.plot(
            0.1 * np.linspace(0, data_trace_hist.size(2) - 1, data_trace_hist.size(2)),
            data_trace_hist[0, j, :].cpu().numpy(),
            color="green",
            linewidth=1.5,
            alpha=0.2,
        )
    plt.ylabel("Sensory input")
    plt.xlabel("Time (ms)")
    plt.title("Neuron trajectories")
    plt.grid()

    # Loop over layers
    for i in range(1, net_depth):
        plt.subplot(net_depth, 1, 1 + i)
        plt.plot(
            0.1 * np.linspace(0, s_hist[i - 1].size(2) - 1, s_hist[i - 1].size(2)),
            s_hist[i - 1].mean(1).mean(0).cpu().numpy(),
            color="blue",
            linewidth=3,
            alpha=0.8,
        )

        for j in range(10):
            plt.plot(
                0.1 * np.linspace(0, s_hist[i - 1].size(2) - 1, s_hist[i - 1].size(2)),
                s_hist[i - 1][0, j, :].cpu().numpy(),
                color="blue",
                linewidth=1.5,
                alpha=0.2,
            )

        plt.xlabel("Time (ms)")
        plt.ylabel("Layer {}".format(i))
        plt.grid()

    fig.tight_layout()
    save_graphic(fig_title, fig)


def plot_apical_trace(
    fig_title,
    n_neu,
    net_depth,
    data_trace_hist,
    va_topdown_hist,
    va_cancelation_hist,
    target_hist,
    s_hist,
):
    fig = plt.figure(figsize=(15, 2 * (net_depth - 1) + 2))
    plot_row_sub = 0

    plt.rcParams.update({"font.size": 12})
    # set figure title
    plt.suptitle(fig_title)
    # plot sensory input
    for j in range(n_neu):
        plt.subplot(net_depth - plot_row_sub, n_neu, j + 1)
        # pick a random input-"neuron" from the visible layer
        ind_neu = j
        plt.plot(
            0.1 * np.linspace(0, data_trace_hist.size(2) - 1, data_trace_hist.size(2)),
            data_trace_hist[0, ind_neu, :].cpu().numpy(),
            color="green",
            linewidth=3,
            alpha=0.5,
        )
        plt.xlabel("Time (ms)")
        if j == 0:
            plt.title("Sensory input")
        plt.grid()

    # Loop over layers
    for i in range(1, net_depth - 1):
        # Loop over randomly picked neurons of each layer
        for j in range(n_neu):
            plt.subplot(net_depth - plot_row_sub, n_neu, n_neu * i + j + 1)
            ind_neu = j
            plt.plot(
                0.1
                * np.linspace(
                    0,
                    va_cancelation_hist[i - 1].size(2) - 1,
                    va_cancelation_hist[i - 1].size(2),
                ),
                va_cancelation_hist[i - 1][0, ind_neu, :].cpu().numpy(),
                color="red",
                linewidth=3,
                alpha=0.5,
                label="Lateral cancelation",
            )
            plt.plot(
                0.1
                * np.linspace(
                    0,
                    va_topdown_hist[i - 1].size(2) - 1,
                    va_topdown_hist[i - 1].size(2),
                ),
                va_topdown_hist[i - 1][0, ind_neu, :].cpu().numpy(),
                color="blue",
                linewidth=3,
                alpha=0.5,
                label="Topdown feedback",
            )
            plt.plot(
                0.1
                * np.linspace(
                    0,
                    va_cancelation_hist[i - 1].size(2) - 1,
                    va_cancelation_hist[i - 1].size(2),
                ),
                (
                    va_cancelation_hist[i - 1][0, ind_neu, :]
                    + va_topdown_hist[i - 1][0, ind_neu, :]
                )
                .cpu()
                .numpy(),
                color="grey",
                linewidth=3,
                alpha=0.5,
                label="Apical potential",
            )
            plt.xlabel("Time (ms)")
            if j == 0:
                plt.ylabel("Layer {}".format(i))
            plt.title("Neuron {}".format(1 + j))
            plt.grid()

    # output layer
    ind_subplot = n_neu * i + j + 2
    for j in range(n_neu):
        plt.subplot(net_depth, n_neu, ind_subplot + j)
        plt.plot(
            0.1 * np.linspace(0, s_hist[-1].size(2) - 1, s_hist[-1].size(2)),
            s_hist[-1][0, j, :].cpu().numpy(),
            color="blue",
            linewidth=3,
            alpha=0.5,
        )
        target_j = []
        for t in target_hist:
            target_j.append(t[0, j, :])
        plt.plot(
            0.1 * np.linspace(0, len(target_j) - 1, len(target_j)),
            target_j,
            color="blue",
            # dotted line
            linestyle="--",
            linewidth=3,
            alpha=0.5,
        )
        plt.xlabel("Time (ms)")
        if j == 0:
            plt.ylabel("Output layer")
        plt.title("Neuron {}".format(1 + j))
        plt.grid()

    fig.tight_layout()
    save_graphic(fig_title, fig)


def plot_apical_distance(fig_title, net_depth, va_topdown_hist, va_cancelation_hist):
    fig = plt.figure(figsize=(15, 2 * (net_depth - 1)))
    plt.rcParams.update({"font.size": 12})
    # set figure title
    plt.suptitle(fig_title)
    for i in range(1, net_depth - 1):
        # plot apical potential through time
        plt.subplot(net_depth - 1, 1, i)
        plt.plot(
            0.1
            * np.linspace(
                0,
                va_cancelation_hist[i - 1].size(2) - 1,
                va_cancelation_hist[i - 1].size(2),
            ),
            ((va_cancelation_hist[i - 1] + va_topdown_hist[i - 1]) ** 2)
            .cpu()
            .numpy()
            .sum(1)
            .mean(0),
            color="grey",
            linewidth=3,
            alpha=0.5,
            label="Apical pot. through time",
        )
        plt.ylabel(r"layer {}".format(i))
        plt.xlabel("Time (ms)")
        plt.title(" Apical potential")
        plt.grid()
    fig.tight_layout()
    save_graphic(fig_title, fig)
