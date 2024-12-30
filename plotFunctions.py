import numpy as np
import matplotlib.pyplot as plt
import os, sys
import torch
import datetime


def save_graphic(filename, fig, run_number):
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    # make sure the directory runs/date_string exists
    os.makedirs(
        r"runs/" + date_string + "/" + r"run_{}".format(run_number), exist_ok=True
    )
    fig.savefig(
        r"runs/"
        + date_string
        + "/"
        + r"run_{}".format(run_number)
        + "/"
        + filename
        + ".png",
        pad_inches=0,
    )


def plot_pyr_int_distance(fig_title, u_p_hist, u_i_hist, run_number):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams.update({"font.size": 12})
    # set figure title
    fig_title += "_pyr_int_distance"
    plt.suptitle(fig_title)
    plt.plot(
        0.1 * np.linspace(0, u_p_hist.size(1) - 1, u_p_hist.size(1)),
        ((u_p_hist - u_i_hist) ** 2).sum(0),
        color="blue",
        linewidth=3,
        alpha=0.8,
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Distance")
    plt.grid()
    plt.tight_layout()
    save_graphic(fig_title, fig, run_number)


def plot_synapse_distance(fig_title, weight_hists, run_number):
    wpf_hist, wpb_hist, wpi_hist, wip_hist = weight_hists
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.rcParams.update({"font.size": 12})
    # set figure title
    fig_title += "_synapse_distance"
    fig.suptitle(fig_title)
    time_step = 0.1 * np.linspace(0, wpf_hist[0].size(2) - 1, wpf_hist[0].size(2))
    sqrd_frob_norm = torch.norm(wpf_hist[1] - wip_hist[0], p="fro", dim=(0, 1)) ** 2
    # plot squared frobenious matrix norm between wpi and wpf and wip and wpb
    axes[0].plot(
        time_step,
        sqrd_frob_norm.numpy(),
        color="grey",
        linewidth=3,
        alpha=0.8,
    )
    axes[0].set_xlabel(r"$||W^{\rm pp, forward} - W^{\rm ip}||^2$")
    axes[0].grid()
    sqrd_frob_norm = torch.norm(wpb_hist[0] - wpi_hist[0], p="fro", dim=(0, 1)) ** 2
    axes[1].plot(
        time_step,
        sqrd_frob_norm.numpy(),
        color="grey",
        linewidth=3,
        alpha=0.8,
    )
    axes[1].set_xlabel(r"$||W^{\rm pp, backward} - W^{\rm pi}||^2$")
    axes[1].grid()

    plt.tight_layout()
    save_graphic(fig_title, fig, run_number)


def plot_synapse_trace(fig_title, net_depth, weight_hists, run_number):
    wpf_hist = weight_hists[0]
    weight_colors = ["red", "blue", "green", "purple"]
    weight_lables = [
        r"$W^{\rm pp, forward}$",
        r"$W^{\rm pp, backward}$",
        r"$W^{\rm pi}$",
        r"$W^{\rm ip}$",
    ]
    fig, axes = plt.subplots(net_depth - 1, 4, figsize=(12, 8))
    # set figure title
    fig_title += "_synapse_trace"
    fig.suptitle(fig_title)
    time_steps = 0.1 * np.linspace(0, wpf_hist[0].size(2) - 1, wpf_hist[0].size(2))

    # forward synapses
    for layer in range(net_depth - 1):
        ax = axes[layer, 0]
        # plot mean across all input synapses
        ax.plot(
            time_steps,
            wpf_hist[layer].mean(0).mean(0).numpy(),
            color="red",
            linewidth=3,
        )
        # plot ten random synapses
        for j in range(5):
            # sample random synapse
            ind_0, ind_1 = np.random.randint(
                wpf_hist[layer].size(0)
            ), np.random.randint(wpf_hist[layer].size(1))
            ax.plot(
                time_steps,
                wpf_hist[layer][ind_0, ind_1, :].numpy(),
                color="red",
                linewidth=1.5,
                alpha=0.2,
            )
        ax.set_title(weight_lables[0])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(r"Layer {}".format(layer))
        ax.grid()

    # all other synapses
    for layer in range(net_depth - 2):
        for w in range(1, 4):
            ax = axes[layer + 1, w]
            # plot mean across all synapses
            ax.plot(
                time_steps,
                weight_hists[w][layer].mean(0).mean(0).numpy(),
                color=weight_colors[w],
                linewidth=3,
                alpha=0.8,
            )

            # plot ten random synapses
            for j in range(5):
                # sample random synapse
                ind_0, ind_1 = np.random.randint(
                    weight_hists[w][layer].size(0)
                ), np.random.randint(weight_hists[w][layer].size(1))
                ax.plot(
                    time_steps,
                    weight_hists[w][layer][ind_0, ind_1, :].numpy(),
                    color=weight_colors[w],
                    linewidth=1.5,
                    alpha=0.2,
                )
            ax.set_xlabel("Time (ms)")
            ax.set_title(weight_lables[w])
            ax.grid()
    plt.tight_layout()
    save_graphic(fig_title, fig, run_number)


def plot_neuron_trace(fig_title, net_depth, data_trace_hist, s_hist, run_number):
    fig = plt.figure(figsize=(5, 2 * net_depth))
    plt.rcParams.update({"font.size": 12})
    # set figure title
    fig_title += "_neuron_trace"
    plt.suptitle(fig_title)
    # plot sensory input
    plt.subplot(net_depth, 1, 1)
    plt.plot(
        0.1 * np.linspace(0, data_trace_hist.size(1) - 1, data_trace_hist.size(1)),
        data_trace_hist.mean(1).numpy(),
        color="green",
        linewidth=3,
        alpha=0.8,
    )
    for j in range(10):
        plt.plot(
            0.1 * np.linspace(0, data_trace_hist.size(1) - 1, data_trace_hist.size(1)),
            data_trace_hist[j, :].numpy(),
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
            0.1 * np.linspace(0, s_hist[i - 1].size(1) - 1, s_hist[i - 1].size(1)),
            s_hist[i - 1].mean(1).numpy(),
            color="blue",
            linewidth=3,
            alpha=0.8,
        )

        for j in range(10):
            plt.plot(
                0.1 * np.linspace(0, s_hist[i - 1].size(1) - 1, s_hist[i - 1].size(1)),
                s_hist[i - 1][j, :].numpy(),
                color="blue",
                linewidth=1.5,
                alpha=0.2,
            )

        plt.xlabel("Time (ms)")
        plt.ylabel("Layer {}".format(i))
        plt.grid()

    save_graphic(fig_title, fig, run_number)


def plot_apical_trace(
    fig_title,
    n_neu,
    net_depth,
    data_trace_hist,
    den_input_hist,
    target_hist,
    s_hist,
    run_number,
):
    fig, axes = plt.subplots(net_depth, n_neu, figsize=(20, 10), squeeze=False)
    fig_title += "_apical_trace"
    plt.suptitle(fig_title, fontsize=16)
    time_steps = 0.1 * np.linspace(
        0, data_trace_hist.shape[0] - 2, data_trace_hist.shape[0] - 1
    )
    va_topdown_hist, va_cancelation_hist, vb_input_hist = den_input_hist
    for j in range(n_neu):
        # Sensory input
        ax = axes[0, j]
        ax.plot(
            time_steps, data_trace_hist[:-1, j], color="green", linewidth=2, alpha=0.8
        )
        ax.set_title(f"Neuron {j+1}", fontsize=12)
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_ylabel("Sensory input", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        for layer in range(1, net_depth - 1):
            ax = axes[layer, j]
            ax.plot(
                time_steps,
                va_topdown_hist[layer - 1][j, :],
                color="red",
                linewidth=2,
                label="Topdown feedback",
            )
            ax.plot(
                time_steps,
                va_cancelation_hist[layer - 1][j, :],
                color="blue",
                linewidth=2,
                label="Lateral cancelation",
            )
            ax.plot(
                time_steps,
                (
                    va_topdown_hist[layer - 1][j, :]
                    + va_cancelation_hist[layer - 1][j, :]
                ),
                color="grey",
                linewidth=2,
                label="Apical potential",
            )
            ax.set_ylabel(f"Layer {layer}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)

        # Output layer
        ax = axes[net_depth - 1, j]
        ax.plot(
            time_steps, s_hist[-1][j, :], color="orange", linewidth=2, label="Output"
        )
        ax.plot(
            time_steps,
            target_hist[:, j],
            color="blue",
            linestyle="--",
            linewidth=2,
            label="Target",
        )
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_ylabel("Output layer", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    save_graphic(fig_title, fig, run_number)


def plot_apical_distance(fig_title, net_depth, va, run_number):
    va_topdown_hist, va_cancelation_hist = va
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams.update({"font.size": 12})
    print(va_cancelation_hist[0].shape)
    print(va_topdown_hist[0].size(1))
    # set figure title
    fig_title += "_apical_distance"
    plt.suptitle(fig_title)
    plt.plot(
        0.1
        * np.linspace(
            0, va_cancelation_hist[0].size(1) - 1, va_cancelation_hist[0].size(1)
        ),
        ((va_cancelation_hist[0] - va_topdown_hist[0]) ** 2).sum(0),
        color="blue",
        linewidth=3,
        alpha=0.8,
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Distance")
    plt.grid()
    plt.tight_layout()
    save_graphic(fig_title, fig, run_number)
