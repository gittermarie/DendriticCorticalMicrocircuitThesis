import torch.optim as optim
import numpy as np

from netClasses import *
from plotFunctions import *
from training_and_eval import *
from config import *
import datetime
import os


def fig_s1(net, config, fig_data, fig_target, train_data, run_number):
    net.to(config.device)
    date_string = datetime.datetime.now().strftime("%Y-%m-%d")
    # make sure the directory runs/date_string/run_number exists
    os.makedirs(
        r"runs/" + date_string + "/" + r"run_{}".format(run_number), exist_ok=True
    )
    with torch.no_grad():
        # before learning self-prediction
        print("---before learning self-prediction---")
        (data_hist, den_in_hist, target_hist, s_hist) = evalrun(
            net, fig_data, fig_target, config
        )
        title = "pre-learning_selfpred"
        plot_apical_trace(
            title,
            5,
            net.depth + 1,
            data_hist,
            den_in_hist,
            target_hist,
            s_hist,
            run_number,
        )

        # learning self-prediction
        print("---learning self-prediction---")
        va, weights_hist, state_hist, n = self_pred_training(net, train_data, config)
        net.save_weights(
            r"weights_selfpred_{}eps.pt".format(n),
            dir=r"runs/" + date_string + "/" + r"run_{}".format(run_number),
        )
        title = r"learning_selfpred({}eps)".format(n * config.batch_size)
        plot_synapse_distance(title, weights_hist, run_number)
        plot_synapse_trace(title, net.depth + 1, weights_hist, run_number)
        plot_apical_distance(title, net.depth + 1, va, run_number)
        plot_pyr_int_distance(title, state_hist[0][-1], state_hist[1][0], run_number)
        # after learning self-prediction
        print("---after learning self-prediction---")
        (data_hist, den_in_hist, target_hist, s_hist) = evalrun(
            net, fig_data, fig_target, config
        )
        title = r"post-learning_selfpred({}eps)".format(n * config.batch_size)
        plot_apical_trace(
            title,
            5,
            net.depth + 1,
            data_hist,
            den_in_hist,
            target_hist,
            s_hist,
            run_number,
        )
        # save the configuration in a file under today's date and run_number
        os.makedirs(
            r"runs/" + date_string + "/" + r"run_{}".format(run_number), exist_ok=True
        )
        with open(
            r"runs/"
            + date_string
            + "/"
            + r"run_{}".format(run_number)
            + "/"
            + "config.txt",
            "w",
        ) as f:
            f.write(str(config))
        # plt.show()


def fig_1(net, config, run_number):
    net.lr_pf = [0.0011875, 0.0005]
    net.tau_weights = 30
    net.lr_ip = [0.0011875]
    net.lr_pi = [0.0005]
    with torch.no_grad():
        net.load_weights("weights/2025-01-16/weights_10000_[30, 20, 10]_val_dist05.pt")
        net.to(config.device)
        net.train()
        n = 1500
        # pre-training evaluation
        data = 2 * torch.rand(1, 30, device=config.device) - 1
        rand_target = 2 * torch.rand(1, 10, device=config.device) - 1
        print("data: ", data)
        print("rand_target: ", rand_target)

        (data_hist, den_in_hist, target_hist, s_hist) = evalrun(
            net, [data, data], [None, rand_target], config
        )
        title = "pre_singtar_{}steps".format(n)
        plot_apical_trace(
            title,
            10,
            net.depth + 1,
            data_hist,
            den_in_hist,
            target_hist,
            s_hist,
            run_number,
        )

        # target training
        apical_dist_hist, pyr_int_dist_hist, val_error_hist = target_training(
            n,
            net,
            data,
            rand_target,
            config.t,
            config.dt,
            config.tau_neu,
            config.device,
        )
        title = "singtar_{}steps".format(n)
        plot_val_hist(
            title,
            net.depth,
            apical_dist_hist,
            pyr_int_dist_hist,
            val_error_hist,
            run_number,
        )

        # post-training evaluation
        (data_hist, den_in_hist, target_hist, s_hist) = evalrun(
            net, [data, data], [None, rand_target], config
        )
        title = "post_singtar_{}steps".format(n)
        plot_apical_trace(
            title,
            10,
            net.depth + 1,
            data_hist,
            den_in_hist,
            target_hist,
            s_hist,
            run_number,
        )
        # plt.show()


def fig_2(net, config, run_number):
    net.to(config.device)
    net.train()
    with torch.no_grad():
        # train non-linear regression task
        global teacherNet
        teacherNet = teacherNet(config.size_tab_teacher, config.k_tab)
        teacherNet.to(config.device)
        try:
            data = create_dataset(
                10000, config.batch_size, net.topology[0], 0, 1, config.device
            )
            target_training(
                15,
                net,
                data,
                teacherNet,
                config.t,
                config.dt,
                config.tau_neu,
                config.device,
            )
        except KeyboardInterrupt:
            pass

        net.save_weights(
            r"weights/2024-12-05/weights_target_{}_{}.pt".format(
                100000 + 1, net.topology
            )
        )

        # plt.show()


if __name__ == "__main__":
    # set seed
    # torch.manual_seed(30)
    config = Config.fig_s1()
    net = dendriticNet(config)
    train_data = create_dataset(
        config.n_samples, config.batch_size, config.size_tab[0], 0, 1
    )
    fig_data = [
        torch.FloatTensor(1, config.size_tab[0]).uniform_(
            config.sample_range[0], config.sample_range[1]
        )
        for _ in range(3)
    ]
    fig_target = [None for _ in range(3)]

    # create training and validation data for fig_2
    # config = Config.fig_2()
    # target_net = teacherNet(config.size_tab, config.k_tab)
    # target_net.to(config.device)
    # train_target = target_net(train_data)

    # check if the run directory exists and if there is already a run_number for today
    date_string = datetime.datetime.now().strftime("%Y-%m-%d")
    run_number = 1
    while os.path.exists(r"runs/" + date_string + "/" + r"run_{}".format(run_number)):
        run_number += 1
    fig_s1(net, config, fig_data, fig_target, train_data, run_number)
    # fig_1(net, config)
    # fig_2(net_2, device)
