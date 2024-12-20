from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import csv
import os
import datetime


class dendriticNet(nn.Module):
    def __init__(
        self,
        T,
        dt,
        batch_size,
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
        freeze_feedback=True,
        init_selfpred=False,
    ):
        super(dendriticNet, self).__init__()

        self.T = T
        self.dt = dt
        self.net_topology = size_tab
        self.net_depth = len(size_tab) - 1
        self.batch_size = batch_size
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
        self.weight_time_const = self.dt / tau_weights
        self.freeze_feedback = freeze_feedback
        self.rho = rho
        self.initw = initw

        # Initialize weights
        self.wpf = nn.ModuleList([])
        self.wpb = nn.ModuleList([])
        self.wpi = nn.ModuleList([])
        self.wip = nn.ModuleList([])

        # Build input weights
        self.wpf.append(
            nn.Linear(self.net_topology[0], self.net_topology[1], bias=False)
        )

        # Build weights for hidden layers
        for i in range(1, self.net_depth):
            self.wpf.append(
                nn.Linear(self.net_topology[i], self.net_topology[i + 1], bias=False)
            )
            torch.nn.init.uniform_(self.wpf[-1].weight, a=-initw, b=initw)
            self.wpb.append(
                nn.Linear(self.net_topology[i + 1], self.net_topology[i], bias=False)
            )
            torch.nn.init.uniform_(self.wpb[-1].weight, a=-initw, b=initw)
            self.wip.append(
                nn.Linear(self.net_topology[i], self.net_topology[i + 1], bias=False)
            )
            torch.nn.init.uniform_(self.wip[-1].weight, a=-initw, b=initw)
            self.wpi.append(
                nn.Linear(self.net_topology[i + 1], self.net_topology[i], bias=False)
            )
            torch.nn.init.uniform_(self.wpi[-1].weight, a=-initw, b=initw)

        # generate self predicting state by hardcoding it (instead of training it)
        if init_selfpred:
            print("self predicting")
            for i in range(len(self.wpi)):
                print("layer {}".format(i))
                self.wip[i].weight.data = (
                    (self.gb + self.glk)
                    / (self.gb + self.glk + self.ga)
                    * self.wpf[i + 1].weight.data.clone()
                )
                self.wpb[i].weight.data = -self.wpi[i].weight.data.clone()

    def updateHist(self, hist, tab, param=False):
        # instantiate a first empty tensor that is on the same device as the net
        if hist == []:
            hist = [torch.empty(0, device=self.which_device()) for _ in range(len(tab))]

        if not param:
            for k in range(len(tab)):
                hist[k] = torch.cat((hist[k], tab[k].unsqueeze(2)), dim=2)
        else:
            for k in range(len(tab)):
                hist[k] = torch.cat((hist[k], tab[k].weight.unsqueeze(2)), dim=2)
        return hist

    def stepper(self, data, s, i, track_va=False, target=None):
        # Prepare lists for derivatives and optionally track apical voltages
        dsdt = [torch.zeros_like(layer) for layer in s]
        didt = [torch.zeros_like(layer) for layer in i]
        if track_va:
            va_topdown = []
            va_cancelation = []

        # Precompute shared values
        rho_data = self.rho(data)
        rho_s = [self.rho(layer) for layer in s]
        rho_i = [self.rho(layer) for layer in i]

        # Compute derivatives for each layer
        for k in range(self.net_depth - 1):
            # Compute basal voltages
            vb = self.wpf[k](rho_data if k == 0 else rho_s[k - 1])
            vi = self.wip[k](rho_s[k])
            va = self.wpi[k](rho_i[k]) + self.wpb[k](rho_s[k + 1])

            if track_va:
                va_topdown.append(self.wpb[k](rho_s[k + 1]))
                va_cancelation.append(self.wpi[k](rho_i[k]))

            # Efficient calculation of i_nudge
            i_nudge = rho_s[k + 1].mean(dim=1, keepdim=True).expand_as(rho_s[k + 1])

            # Derivatives for somatic and interneuron compartments
            dsdt[k] = (
                -self.glk * s[k]
                + self.gb * (vb - s[k])
                + self.ga * (va - s[k])
                + self.noise * torch.randn_like(s[k], device=s[k].device)
            )
            didt[k] = (
                -self.glk * i[k]
                + self.gd * (vi - i[k])
                + self.gsom * (i_nudge - i[k])
                + self.noise * torch.randn_like(i[k], device=i[k].device)
            )

        # Derivative for the output layer
        vb = self.wpf[-1](rho_s[-2])
        dsdt[-1] = (
            -self.glk * s[-1]
            + self.gb * (vb - s[-1])
            + self.noise * torch.randn_like(s[-1])
        )
        if target is not None:
            dsdt[-1] += self.gsom * (target - s[-1])

        # Update somatic and interneuron potentials
        for k in range(self.net_depth - 1):
            s[k] += self.dt * dsdt[k]
            i[k] += self.dt * didt[k]
        s[-1] += self.dt * dsdt[-1]

        # Return updated potentials and optionally apical inputs
        if track_va:
            return s, i, [va_topdown, va_cancelation]
        return s, i

    def save_weights(self, filename):
        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d")
        # make sure the directory graphics/date_string exists
        os.makedirs(r"weights/" + date_string, exist_ok=True)

        # Save state_dicts of the ModuleLists
        torch.save(
            {
                "wpf": self.wpf.state_dict(),
                "wpb": self.wpb.state_dict(),
                "wip": self.wip.state_dict(),
                "wpi": self.wpi.state_dict(),
            },
            "weights/" + date_string + "/" + filename,
        )

    def load_weights(self, file_path):
        # Load state_dicts into the ModuleLists
        checkpoint = torch.load(file_path)
        self.wpf.load_state_dict(checkpoint["wpf"])
        self.wpb.load_state_dict(checkpoint["wpb"])
        self.wip.load_state_dict(checkpoint["wip"])
        self.wpi.load_state_dict(checkpoint["wpi"])

    def initHidden(self, **kwargs):
        s = []
        i = []

        # initialize the membrane potential for neurons in the hidden layers
        for k in range(1, self.net_depth):
            s.append(torch.zeros(self.batch_size, self.net_topology[k]))
            # there is as many interneurons as in the next layer
            i.append(torch.zeros(self.batch_size, self.net_topology[k + 1]))

        # initialize the membrane potential for the output layer
        s.append(torch.zeros(self.batch_size, self.net_topology[-1]))

        if "device" in kwargs:
            for k in range(len(s)):
                s[k] = s[k].to(kwargs["device"])
            for k in range(len(i)):
                if i[k] is not None:
                    i[k] = i[k].to(kwargs["device"])
        return s, i

    def computeGradients(self, s, i, data, target=None):
        # Precompute shared values
        rho_data = self.rho(data)
        rho_s = [self.rho(layer) for layer in s]
        rho_i = [self.rho(layer) for layer in i]

        # Initialize gradients
        dWpf = [torch.zeros_like(w.weight) for w in self.wpf]
        dWpi = [torch.zeros_like(w.weight) for w in self.wpi]
        dWpb = [torch.zeros_like(w.weight) for w in self.wpb]
        dWip = [torch.zeros_like(w.weight) for w in self.wip]

        # Compute gradients for each layer
        for k in range(self.net_depth - 1):
            # Precompute apical inputs
            vb = self.wpf[k](rho_data if k == 0 else rho_s[k - 1])
            vi = self.wip[k](rho_s[k])
            va = self.wpi[k](rho_i[k]) + self.wpb[k](rho_s[k + 1])

            # Basal gradient
            dWpf[k] = torch.einsum(
                "bi,bj->bji",
                rho_data if k == 0 else rho_s[k - 1],
                self.gb * (vb - s[k]),
            )

            # Interneuron gradient
            dWip[k] = torch.einsum("bi,bj->bji", rho_s[k], self.gd * (vi - i[k]))

            # Apical gradients
            dWpi[k] = torch.einsum("bi,bj->bji", rho_i[k], self.ga * (va - s[k]))
            dWpb[k] = torch.einsum("bi,bj->bji", rho_s[k + 1], self.ga * (va - s[k]))

        # Output layer gradients
        vb = self.wpf[-1](rho_s[-2])
        dWpf[-1] = torch.einsum("bi,bj->bji", rho_s[-2], self.gb * (vb - s[-1]))
        if target is not None:
            dWpf[-1] += torch.einsum(
                "bi,bj->bji", rho_s[-2], self.gsom * (target - s[-1])
            )

        # Return computed gradients
        return dWpf, dWpi, dWpb, dWip

    def updateWeights(self, data, s, i, target=None):

        dWpf, dWpi, dWpb, dWip = self.computeGradients(s, i, data, target)

        # Update weights for each layer
        for k in range(self.net_depth - 1):
            # Perform in-place updates to reduce memory usage
            self.wpf[k].weight -= (
                self.lr_pf[k] * self.weight_time_const * dWpf[k].mean(dim=0)
            )
            self.wpi[k].weight -= (
                self.lr_pi[k] * self.weight_time_const * dWpi[k].mean(dim=0)
            )
            self.wpb[k].weight -= (
                self.lr_pb[k] * self.weight_time_const * dWpb[k].mean(dim=0)
            )
            self.wip[k].weight -= (
                self.lr_ip[k] * self.weight_time_const * dWip[k].mean(dim=0)
            )

        # Output layer updates
        self.wpf[-1].weight -= (
            self.lr_pf[-1] * self.weight_time_const * dWpf[-1].mean(dim=0)
        )

    def extra_repr(self):

        return f"T-{self.T}_dt-{self.dt}_topology-{'-'.join(map(str,self.net_topology))}_lrpp-{'-'.join(map(str,self.lr_pf))}_lrpi-{'-'.join(map(str,self.lr_pi))}_lrip-{'-'.join(map(str,self.lr_ip))}_ga-{self.ga}_gb-{self.gb}_gd-{self.gd}_glk-{self.glk}_gsom-{self.gsom}_rho-{self.rho.__name__}_initw-{self.initw}"

    def which_device(self):
        return next(self.parameters()).device


class teacherNet(nn.Module):
    def __init__(self, size_tab, k_tab):
        super(teacherNet, self).__init__()

        self.net_topology = size_tab

        self.gamma = 0.1
        self.beta = 1
        self.theta = 3
        self.k_tab = k_tab

        w = nn.ModuleList([])

        # Build weights
        for i in range(len(self.net_topology) - 1):
            w.append(
                nn.Linear(self.net_topology[i], self.net_topology[i + 1], bias=False)
            )
            torch.nn.init.uniform_(w[i].weight, a=-1, b=1)

        self.w = w

    def rho(self, x):
        return self.gamma * torch.log(1 + torch.exp(self.beta * (x - self.theta)))

    def forward(self, x):
        a = x
        for i in range(len(self.w)):
            a = self.rho(self.k_tab[i] * self.w[i](a))

        return a
