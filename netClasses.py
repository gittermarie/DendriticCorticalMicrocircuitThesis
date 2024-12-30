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
        device=torch.device("cpu"),
    ):
        super(dendriticNet, self).__init__()

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
        self.rho = rho
        self.device = device
        self.s = [torch.zeros(batch_size, size, device=device) for size in size_tab[1:]]
        self.i = [torch.zeros(batch_size, size, device=device) for size in size_tab[2:]]

        # Initialize weights
        self.wpf = nn.ModuleList([])
        self.wpb = nn.ModuleList([])
        self.wpi = nn.ModuleList([])
        self.wip = nn.ModuleList([])

        # Build input weights
        self.wpf.append(nn.Linear(size_tab[0], size_tab[1], bias=False))

        # Build weights for hidden layers
        for i in range(1, self.net_depth):
            self.wpf.append(nn.Linear(size_tab[i], size_tab[i + 1], bias=False))
            torch.nn.init.uniform_(self.wpf[-1].weight, a=-initw, b=initw)
            self.wpb.append(nn.Linear(size_tab[i + 1], size_tab[i], bias=False))
            torch.nn.init.uniform_(self.wpb[-1].weight, a=-initw, b=initw)
            self.wip.append(nn.Linear(size_tab[i], size_tab[i + 1], bias=False))
            torch.nn.init.uniform_(self.wip[-1].weight, a=-initw, b=initw)
            self.wpi.append(nn.Linear(size_tab[i + 1], size_tab[i], bias=False))
            torch.nn.init.uniform_(self.wpi[-1].weight, a=-initw, b=initw)

        self.dWpf = [torch.zeros_like(w.weight, device=device) for w in self.wpf]
        self.dWpi = [torch.zeros_like(w.weight, device=device) for w in self.wpi]
        self.dWpb = [torch.zeros_like(w.weight, device=device) for w in self.wpb]
        self.dWip = [torch.zeros_like(w.weight, device=device) for w in self.wip]

        self.dsdt = [torch.zeros_like(s, device=device) for s in self.s]
        self.didt = [torch.zeros_like(i, device=device) for i in self.i]
        self.i_nudge = [torch.zeros_like(i, device=device) for i in self.i]

    def to(self, device):
        super().to(device)
        self.device = device
        self.s = [s.to(device) for s in self.s]
        self.i = [i.to(device) for i in self.i]
        self.dsdt = [dsdt.to(device) for dsdt in self.dsdt]
        self.didt = [didt.to(device) for didt in self.didt]
        self.i_nudge = [i_nudge.to(device) for i_nudge in self.i_nudge]
        self.dWpf = [dWpf.to(device) for dWpf in self.dWpf]
        self.dWpi = [dWpi.to(device) for dWpi in self.dWpi]
        self.dWpb = [dWpb.to(device) for dWpb in self.dWpb]
        self.dWip = [dWip.to(device) for dWip in self.dWip]
        # move module lists to device
        self.wpf = self.wpf.to(device)
        self.wpb = self.wpb.to(device)
        self.wpi = self.wpi.to(device)
        self.wip = self.wip.to(device)
        return self

    def stepper(self, data, target=None):
        # prepare lists for derivatives and apical voltages
        va_topdown = []
        va_cancelation = []

        # precompute shared values
        rho_s = [self.rho(s) for s in self.s]
        rho_i = [self.rho(i) for i in self.i]
        rho_data = self.rho(data)

        # Compute voltages and derivatives of the hidden layers:
        for k in range(self.net_depth - 1):
            vb = self.wpf[k](rho_data if k == 0 else rho_s[k - 1])
            vi = self.wip[k](rho_s[k])
            va = self.wpi[k](rho_i[k]) + self.wpb[k](rho_s[k + 1])

            va_topdown.append(self.wpb[k](rho_s[k + 1]))
            va_cancelation.append(self.wpi[k](rho_i[k]))

            # compute i_nudge which is the average of the somatic activation of the next layer and should have the same shape as the interneuron activation
            self.i_nudge[k] = (
                self.s[k + 1].mean(dim=0).unsqueeze(0).repeat(self.batch_size, 1)
            )
            # Compute total derivative of somatic voltage (Eq. 1)
            self.dsdt[k] = (
                -self.glk * self.s[k]
                + self.gb * (vb - self.s[k])
                + self.ga * (va - self.s[k])
                + self.noise * torch.randn_like(self.s[k], device=self.device)
            )
            # Compute total derivative of the interneuron (Eq. 2)
            self.didt[k] = (
                -self.glk * self.i[k]
                + self.gd * (vi - self.i[k])
                + self.gsom * (self.i_nudge[k] - self.i[k])
                + self.noise * torch.randn_like(self.i[k], device=self.device)
            )

        # Compute derivative of the output layer
        vb = self.wpf[-1](rho_s[-2])
        self.dsdt[-1] = (
            -self.glk * self.s[-1]
            + self.gb * (vb - self.s[-1])
            + self.noise * torch.randn_like(self.s[-1], device=self.device)
        )
        # Nudge the derivative of the output neuron in the direction of the target
        if target is not None:
            self.dsdt[-1] = self.dsdt[-1] + self.gsom * (target - self.s[-1])
        # Update the values of the neurons in the hidden layers by adding the time step times the derivative
        for k in range(self.net_depth - 1):
            self.s[k] += self.dt * self.dsdt[k]
            self.i[k] += self.dt * self.didt[k]
        # Update the values of the output layer
        self.s[-1] = self.dt * self.dsdt[-1]
        # return the somatic and interneuron membrane potentials and optionally the apical inputs

        return [va_topdown, va_cancelation]

    def computeGradients(self, data):
        rho_s = [self.rho(s) for s in self.s]
        rho_i = [self.rho(i) for i in self.i]
        rho_data = self.rho(data)

        for k in range(self.net_depth - 1):
            vb = self.wpf[k](rho_data if k == 0 else rho_s[k - 1])
            vi = self.wip[k](rho_s[k])
            va = self.wpi[k](rho_i[k]) + self.wpb[k](rho_s[k + 1])
            vtd = self.wpb[k](rho_s[k + 1])
            # voltage of the basal compartment when taking into account the dendritic attenuation factors
            vbhat = self.gb / (self.gb + self.glk + self.ga) * vb
            vihat = self.gd / (self.gd + self.glk) * vi

            self.dWpf[k].copy_(
                torch.mm(
                    torch.transpose(rho_s[k] - self.rho(vbhat), 0, 1),
                    (rho_data if k == 0 else rho_s[k - 1]),
                )
                / self.batch_size
            )
            self.dWip[k].copy_(
                torch.mm(
                    torch.transpose(rho_i[k] - self.rho(vihat), 0, 1),
                    rho_s[k],
                )
                / self.batch_size
            )
            self.dWpi[k].copy_(
                torch.mm(torch.transpose(-va, 0, 1), rho_i[k]) / self.batch_size
            )
            self.dWpb[k].copy_(
                torch.mm(
                    torch.transpose(rho_s[k] - self.rho(vtd), 0, 1),
                    rho_s[k + 1],
                )
                / self.batch_size
            )

        # for the output layer
        vb = self.wpf[-1](rho_s[-2])
        vbhat = self.gb / (self.gb + self.glk + self.ga) * vb
        self.dWpf[-1].copy_(
            torch.mm(
                torch.transpose(rho_s[-1] - self.rho(vbhat), 0, 1),
                rho_s[-2],
            )
            / self.batch_size
        )

    def updateWeights(self, data):

        self.computeGradients(data)

        for k in range(self.net_depth - 1):
            self.wpf[k].weight += (
                self.lr_pf[k] * (self.dt / self.tau_weights) * self.dWpf[k]
            )
            self.wpb[k].weight += (
                self.lr_pb[k] * (self.dt / self.tau_weights) * self.dWpb[k]
            )
            self.wpi[k].weight += (
                self.lr_pi[k] * (self.dt / self.tau_weights) * self.dWpi[k]
            )
            self.wip[k].weight += (
                self.lr_ip[k] * (self.dt / self.tau_weights) * self.dWip[k]
            )
        self.wpf[-1].weight += (
            self.lr_pf[-1] * (self.dt / self.tau_weights) * self.dWpf[-1]
        )

    def updateHist(self, hist, tab, param=False):
        # instantiate a first empty tensor that is on the same device as the net
        if hist == []:
            hist = [torch.empty(0, device=self.device) for _ in range(len(tab))]

        if not param:
            for k in range(len(tab)):
                hist[k] = torch.cat((hist[k], tab[k].unsqueeze(2)), dim=2)
        else:
            for k in range(len(tab)):
                hist[k] = torch.cat((hist[k], tab[k].weight.unsqueeze(2)), dim=2)
        return hist

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

    def extra_repr(self):
        return f"dt-{self.dt}_topology-{'-'.join(map(str,self.net_topology))}_lrpp-{'-'.join(map(str,self.lr_pp))}_lrpi-{'-'.join(map(str,self.lr_pi))}_lrip-{'-'.join(map(str,self.lr_ip))}_ga-{self.ga}_gb-{self.gb}_gd-{self.gd}_glk-{self.glk}_gsom-{self.gsom}_rho-{self.rho.__name__}"


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
