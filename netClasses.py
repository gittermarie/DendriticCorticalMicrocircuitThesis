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
        lr_pp,
        lr_ip,
        lr_pi,
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
        self.lr_pp = lr_pp
        self.lr_ip = lr_ip
        self.lr_pi = lr_pi
        self.ga = ga
        self.gb = gb
        self.gd = gd
        self.glk = glk
        self.gsom = gsom
        self.noise = noise
        self.tau_weights = tau_weights
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
        if hist == []:
            hist = [torch.empty(0) for _ in range(len(tab))]

        if not param:
            for k in range(len(tab)):
                hist[k] = torch.cat((hist[k], tab[k].unsqueeze(2)), dim=2)
        else:
            for k in range(len(tab)):
                hist[k] = torch.cat((hist[k], tab[k].weight.unsqueeze(2)), dim=2)
        return hist

    def forward(self, data, s, i, track_va=False, target=None):
        for t in range(self.T):
            if track_va:
                s, i, va = self.stepper(data, s, i, True, target)
            else:
                s, i = self.stepper(data, s, i, track_va, target)
        if track_va:
            return s, i, va
        return s, i

    def stepper(self, data, s, i, track_va=False, target=None):
        # derivatives
        dsdt = []
        didt = []
        # apical voltage
        if track_va:
            va_topdown = []
            va_cancelation = []

        # Compute voltages and derivatives of the hidden layers:
        for k in range(self.net_depth - 1):
            # Compute basal voltages
            if k == 0:
                vb = self.wpf[k](self.rho(data))
            else:
                vb = self.wpf[k](self.rho(s[k - 1]))
            # Compute interneuron basal voltages
            vi = self.wip[k](self.rho(s[k]))
            # Compute apical voltages
            va = self.wpi[k](self.rho(i[k])) + self.wpb[k](self.rho(s[k + 1]))
            if track_va:
                va_topdown.append(self.wpb[k](self.rho(s[k + 1])))
                va_cancelation.append(self.wpi[k](self.rho(i[k])))
            # create a matrix of shape s[k+1] x i[k] filled with 1/self.net_topology[k+1]
            i_nudge = torch.matmul(
                torch.full(
                    (self.net_topology[k + 2], self.net_topology[k + 2]),
                    1 / self.net_topology[k + 2],
                ),
                self.rho(s[k + 1]).squeeze(0),
            ).unsqueeze(0)
            # Compute total derivative of somatic voltage (Eq. 1)
            dsdt.append(
                -self.glk * s[k]
                + self.gb * (vb - s[k])
                + self.ga * (va - s[k])
                + self.noise * torch.randn_like(s[k])
            )
            # Compute total derivative of the interneuron (Eq. 2)
            didt.append(
                -self.glk * i[k]
                + self.gd * (vi - i[k])
                + self.gsom * (i_nudge - i[k])
                + self.noise * torch.randn_like(i[k])
            )

        # Compute derivative of the output layer
        vb = self.wpf[-1](self.rho(s[-2]))
        dsdt.append(
            -self.glk * s[-1]
            + self.gb * (vb - s[-1])
            + self.noise * torch.randn_like(s[-1])
        )
        # Nudge the derivative of the output neuron in the direction of the target
        if target is not None:
            dsdt[-1] = dsdt[-1] + self.gsom * (target - s[-1])
        # Update the values of the neurons in the hidden layers by adding the time step times the derivative
        for k in range(self.net_depth - 1):
            s[k] += self.dt * dsdt[k]
            i[k] += self.dt * didt[k]
        # Update the values of the output layer
        s[-1] = self.dt * dsdt[-1]
        # return the somatic and interneuron membrane potentials and optionally the apical inputs
        if not track_va:
            return s, i
        else:
            return s, i, [va_topdown, va_cancelation]

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

    def computeGradients(self, data, s, i):
        gradwpf = []
        gradwpb = []
        gradwip = []
        gradwpi = []

        for k in range(self.net_depth - 1):
            # for the first layer
            if k == 0:
                input = data
            else:
                input = s[k - 1]
                # voltage of the basal compartment is equal to activation of input times forward weights
            vb = self.wpf[k](self.rho(input))
            vi = self.wip[k](self.rho(s[k]))
            va = self.wpi[k](self.rho(i[k])) + self.wpb[k](self.rho(s[k + 1]))
            # voltage of the basal compartment when taking into account the dendritic attenuation factors
            vbhat = self.gb / (self.gb + self.glk + self.ga) * vb
            vihat = self.gd / (self.gd + self.glk) * vi
            vtdhat = self.wpb[k](self.rho(s[k + 1]))
            gradwpf.append(
                (1 / self.batch_size)
                * (
                    torch.mm(
                        # gradients are equal to somatic activation minus basal activation times
                        torch.transpose(self.rho(s[k]) - self.rho(vbhat), 0, 1),
                        self.rho(input),
                    )
                )
            )
            gradwip.append(
                (1 / self.batch_size)
                * (
                    torch.mm(
                        torch.transpose(self.rho(i[k]) - self.rho(vihat), 0, 1),
                        self.rho(s[k]),
                    )
                )
            )
            gradwpi.append(
                (1 / self.batch_size)
                * (torch.mm(torch.transpose(-va, 0, 1), self.rho(i[k])))
            )
            gradwpb.append(
                (1 / self.batch_size)
                * (
                    torch.mm(
                        torch.transpose(self.rho(s[k]) - self.rho(vtdhat), 0, 1),
                        self.rho(s[k + 1]),
                    )
                )
            )

        # for the output layer
        vb = self.wpf[-1](self.rho(s[-2]))
        vbhat = self.gb / (self.gb + self.glk + self.ga) * vb
        gradwpf.append(
            (1 / self.batch_size)
            * (
                torch.mm(
                    # gradients are equal to somatic activation minus basal activation times
                    torch.transpose(self.rho(s[-1]) - self.rho(vbhat), 0, 1),
                    self.rho(s[-2]),
                )
            )
        )
        return gradwpf, gradwpb, gradwpi, gradwip

    def updateWeights(self, data, s, i, selfpredict=False, freeze_feedback=False):

        gradwpf, gradwpb, gradwpi, gradwip = self.computeGradients(data, s, i)

        if not selfpredict:
            for k in range(len(self.wpf)):
                self.wpf[k].weight += (
                    self.lr_pp[k] * (self.dt / self.tau_weights) * gradwpf[k]
                )
                if not freeze_feedback:
                    for k in range(len(self.wpb)):
                        self.wpb[k].weight += (
                            self.lr_pp[k] * (self.dt / self.tau_weights) * gradwpb[k]
                        )

        for k in range(len(self.wpi)):
            self.wpi[k].weight += (
                self.lr_pi[k] * (self.dt / self.tau_weights) * gradwpi[k]
            )
            self.wip[k].weight += (
                self.lr_ip[k] * (self.dt / self.tau_weights) * gradwip[k]
            )

    def extra_repr(self):

        return f"T-{self.T}_dt-{self.dt}_topology-{'-'.join(map(str,self.net_topology))}_lrpp-{'-'.join(map(str,self.lr_pp))}_lrpi-{'-'.join(map(str,self.lr_pi))}_lrip-{'-'.join(map(str,self.lr_ip))}_ga-{self.ga}_gb-{self.gb}_gd-{self.gd}_glk-{self.glk}_gsom-{self.gsom}_rho-{self.rho.__name__}_initw-{self.initw}"


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
