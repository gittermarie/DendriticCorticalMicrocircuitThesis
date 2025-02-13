from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import os
import datetime


class dendriticNet(nn.Module):
    def __init__(self, config):
        super(dendriticNet, self).__init__()
        self.config = config
        self.topology = config.size_tab
        self.depth = len(config.size_tab) - 1
        self.device = config.device
        # Initialize state vectors
        self.initialize_states(config.batch_size)
        self.initialize_weights()

    def initialize_weights(self):
        self.wpf = nn.ModuleList([])
        self.wpb = nn.ModuleList([])
        self.wpi = nn.ModuleList([])
        self.wip = nn.ModuleList([])

        # Build input weights
        self.wpf.append(nn.Linear(self.topology[0], self.topology[1], bias=False))

        # Build weights for hidden layers
        for i in range(1, self.depth):
            self.wpf.append(
                nn.Linear(self.topology[i], self.topology[i + 1], bias=False)
            )
            torch.nn.init.uniform_(
                self.wpf[-1].weight, a=-self.config.initw, b=self.config.initw
            )
            self.wpb.append(
                nn.Linear(self.topology[i + 1], self.topology[i], bias=False)
            )
            torch.nn.init.uniform_(
                self.wpb[-1].weight, a=-self.config.initw, b=self.config.initw
            )
            self.wip.append(
                nn.Linear(self.topology[i], self.topology[i + 1], bias=False)
            )
            torch.nn.init.uniform_(
                self.wip[-1].weight, a=-self.config.initw, b=self.config.initw
            )
            self.wpi.append(
                nn.Linear(self.topology[i + 1], self.topology[i], bias=False)
            )
            torch.nn.init.uniform_(
                self.wpi[-1].weight, a=-self.config.initw, b=self.config.initw
            )

        # Initialize weight derivatives
        self.dwpf = [torch.zeros_like(w.weight, device=self.device) for w in self.wpf]
        self.dwpi = [torch.zeros_like(w.weight, device=self.device) for w in self.wpi]
        self.dwpb = [torch.zeros_like(w.weight, device=self.device) for w in self.wpb]
        self.dwip = [torch.zeros_like(w.weight, device=self.device) for w in self.wip]

    def initialize_states(self, batch_size):
        """Initialize or reset the state vectors based on the current batch size."""
        self.u_p = [
            torch.zeros(batch_size, size, device=self.device)
            for size in self.topology[1:]
        ]
        self.u_i = [
            torch.zeros(batch_size, size, device=self.device)
            for size in self.topology[2:]
        ]
        self.r_p = [torch.zeros_like(u_p, device=self.device) for u_p in self.u_p]
        self.r_i = [torch.zeros_like(u_i, device=self.device) for u_i in self.u_i]
        self.du_p = [torch.zeros_like(u_p, device=self.device) for u_p in self.u_p]
        self.du_i = [torch.zeros_like(u_i, device=self.device) for u_i in self.u_i]

    def compute_nudging(self, u, u_trgt):
        # excitatory and inhibitory synaptic reversal potentials
        e_exc = 1.0
        e_inh = -1.0
        # g^P_{exc,N} = g_{som} \frac{u^{trgt}_{N} - E_{inh}}{E_{exc} - E_{inh}}
        g_exc = self.config.gsom * (u_trgt - e_inh) / (e_exc - e_inh)
        # g^P_{inh,N} = -g_{som} \frac{u^{trgt}_{N} - E_{exc}}{E_{exc} - E_{inh}}
        g_inh = -self.config.gsom * (u_trgt - e_exc) / (e_exc - e_inh)
        # i^P_{N}(t) = g^P_{exc,N} (E_{exc} - u^P_{N}(t)) + g^P_{inh,N} (E_{inh} - u^P_{N}(t))
        i_nudge = g_exc * (e_exc - u) + g_inh * (e_inh - u)
        return i_nudge

    def stepper(self, data, target=None, plasticity=False):
        """
        Update neuron statesand synapses based on input data and compute apical potentials.

        Args:
            data (torch.Tensor): Input data with shape [batch_size, input_size].
            target (torch.Tensor, optional): Target values for the output layer.
            plasticity (bool, optional): Whether to update the weights.

        Returns:
            list: [va_topdown, va_cancelation] apical potentials.
        """
        batch_size = data.size(0)

        if self.u_p[0].size(0) != batch_size or self.u_i[0].size(0) != batch_size:
            self.initialize_states(batch_size)

        va_topdown = []
        va_cancelation = []
        vb_input = []

        # Loop over layers
        for k in range(self.depth - 1):
            input_rho = self.config.rho(data) if k == 0 else self.r_p[k - 1]
            # v^P_{B,k}(t) = W^{PP}_{k,k-1} \phi(u^P_{k-1}(t)) (3)
            vb = self.wpf[k](input_rho)
            # v^I_{k}(t) = W^{IP}_{k,k} \phi(u^P_{k}(t)) (5)
            vi = self.wip[k](self.r_p[k])
            # v^P_{A,k}(t) = W^{PP}_{k,k+1} \phi(u^P_{k+1}(t))+ W^{PI}_{k,k} \phi(u^I_{k}(t)) (4)
            va = self.wpb[k](self.r_p[k + 1]) + self.wpi[k](self.r_i[k])

            va_topdown.append(self.wpb[k](self.r_p[k + 1]))
            va_cancelation.append(self.wpi[k](self.r_i[k]))
            vb_input.append(vb)

            # interneuron nudging calculation
            # i_nudge = self.compute_nudging(self.u_i[k], self.u_p[k + 1])
            i_nudge = self.config.gsom * (self.u_p[k + 1] - self.u_i[k])

            # Update derivatives
            # \frac{d}{dt}u^P_{k}(t) = -g_{lk}u^P_{k}(t) + g_{b}(v^P_{B,k}(t) - u^P_{k}(t)) + g_{a}(v^P_{A,k}(t) - u^P_{k}(t)) + \sigma\xi(t) (1)
            self.du_p[k] = (
                -self.config.glk * self.u_p[k]
                + self.config.gb * (vb - self.u_p[k])
                + self.config.ga * (va - self.u_p[k])
                + self.config.noise * torch.randn_like(self.u_p[k], device=self.device)
            )

            # \frac{d}{dt}u^I_{k}(t) = -g_{lk}u^I_{k}(t) + g_{d}(v^I_{k}(t) - u^I_{k}(t)) + i^I_{k}(t) + \sigma\xi(t) (2)
            self.du_i[k] = (
                -self.config.glk * self.u_i[k]
                + self.config.gd * (vi - self.u_i[k])
                + i_nudge
                + self.config.noise * torch.randn_like(self.u_i[k], device=self.device)
            )

        # Compute derivative of the output layer
        # v^P_{B,k}(t) = W^{PP}_{k,k-1} \phi(u^P_{k-1}(t)) (3)
        vb = self.wpf[-1](self.r_p[-2])
        vb_input.append(vb)
        # nudge output towards target
        if target is not None:
            # i_nudge = self.compute_nudging(self.u_p[-1], target)
            i_nudge = self.config.gsom * (target - self.u_p[-1])
        else:
            i_nudge = torch.zeros_like(self.u_p[-1])

        # \frac{d}{dt}u^P_{k}(t) = -g_{lk}u^P_{k}(t) + g_{b}(v^P_{B,k}(t) - u^P_{k}(t)) + i^P_{N}(t) + \sigma\xi(t)
        self.du_p[-1].copy_(
            -self.config.glk * self.u_p[-1]
            + self.config.gb * (vb - self.u_p[-1])
            + i_nudge
            + self.config.noise * torch.randn_like(self.u_p[-1], device=self.device)
        )

        # Update the values of the neurons in the hidden layers by adding the time step times the derivative
        for k in range(self.depth - 1):
            self.u_p[k].add_(self.config.dt * self.du_p[k])
            self.u_i[k].add_(self.config.dt * self.du_i[k])

        self.u_p[-1].add_(self.config.dt * self.du_p[-1])

        # update the rates
        self.r_p = [self.config.rho(u_p) for u_p in self.u_p]
        self.r_i = [self.config.rho(u_i) for u_i in self.u_i]

        return [va_topdown, va_cancelation, vb_input]

    def updateWeights(self, data):
        for k in range(self.depth - 1):
            # v^P_{B,k}(t) = W^{PP}_{k,k-1} \phi(u^P_{k-1}(t)) (3)
            vb = self.wpf[k](self.config.rho(data) if k == 0 else self.r_p[k - 1])
            # v^I_{k}(t) = W^{IP}_{k,k} \phi(u^P_{k}(t)) (5)
            vi = self.wip[k](self.r_p[k])
            # v^P_{A,k}(t) = W^{PP}_{k,k+1} \phi(u^P_{k+1}(t))+ W^{PI}_{k,k} \phi(u^I_{k}(t)) (4)
            va = self.wpi[k](self.r_i[k]) + self.wpb[k](self.r_p[k + 1])

            # \hat{v}^P_{TD,k} = W_{k,k+1} r_p_{k+1}
            vtd = self.wpb[k](self.r_p[k + 1])

            # voltage of the basal compartment when taking into account the dendritic attenuation factors
            # \hat{v}^P_{B,k} = \frac{g_b}{g_{lk} + g_b + g_a} v^P_{B,k}
            vbhat = (
                self.config.gb
                / (self.config.glk + self.config.gb + self.config.ga)
                * vb
            )
            # \hat{v}^I_k = \frac{g_d}{g_{lk} + g_d} v^I_k
            vihat = self.config.gd / (self.config.gd + self.config.glk) * vi

            # \frac{d}{dt}W^{PP}_{k,k-1} = \nabla^{PP}_{k,k-1}(\phi(u^P_k)-\phi(\hat{v}^P_{B,k}))(r^P_{k-1})^T (7)
            self.dwpf[k].copy_(
                self.config.lr_pf[k]
                * torch.mm(
                    torch.transpose(self.r_p[k] - self.config.rho(vbhat), 0, 1),
                    (self.config.rho(data)) if k == 0 else self.r_p[k - 1],
                )
            )
            # \frac{d}{dt}W^{IP}_{k,k} = \nabla^{IP}_{k,k}(\phi(u^I_k)-\phi(\hat{v}^I_k))(r^I_k)^T (8)
            self.dwip[k].copy_(
                self.config.lr_ip[k]
                * torch.mm(
                    torch.transpose(self.r_i[k] - self.config.rho(vihat), 0, 1),
                    self.r_p[k],
                )
            )
            # \frac{d}{dt}W^{PI}_{k,k} = \nabla^{PI}_{k,k}(v_{rest}-v^P_{A,k}))(r^I_k)^T (9)
            self.dwpi[k].copy_(
                self.config.lr_pi[k] * torch.mm(torch.transpose(-va, 0, 1), self.r_i[k])
            )
            # \frac{d}{dt}W^{PP}_{k,k+1} = \nabla^{PP}_{k,k+1}(\phi(u^P_k)-\phi(\hat{v}^P_{TD,k}))(r^P_{k+1})^T (10)
            self.dwpb[k].copy_(
                self.config.lr_pb[k]
                * torch.mm(
                    torch.transpose(self.r_p[k] - self.config.rho(vtd), 0, 1),
                    self.r_p[k + 1],
                )
            )

        # for the output layer
        # v^P_{B,N}(t) = W^{PP}_{k,k-1} \phi(u^P_{k-1}(t)) (3)
        vb = self.wpf[-1](self.r_p[-2])
        # \hat{v}^P_{B,k} = \frac{g_b}{g_{lk} + g_b + g_a} v^P_{B,k}
        vbhat = (
            self.config.gb / (self.config.glk + self.config.gb + self.config.ga) * vb
        )
        # \frac{d}{dt}W^{PP}_{k,k-1} = \nabla^{PP}_{k,k-1}(\phi(u^P_k)-\phi(\hat{v}^P_{B,k}))(r^P_{k-1})^T (7)
        self.dwpf[-1].copy_(
            self.config.lr_pf[-1]
            * torch.mm(
                torch.transpose(self.r_p[-1] - self.config.rho(vbhat), 0, 1),
                self.r_p[-2],
            )
        )

        for k in range(self.depth - 1):
            self.wpf[k].weight.add_(
                (self.config.dt / self.config.tau_weights) * self.dwpf[k]
            )
            self.wpb[k].weight.add_(
                (self.config.dt / self.config.tau_weights) * self.dwpb[k]
            )
            self.wpi[k].weight.add_(
                (self.config.dt / self.config.tau_weights) * self.dwpi[k]
            )
            self.wip[k].weight.add_(
                (self.config.dt / self.config.tau_weights) * self.dwip[k]
            )
        self.wpf[-1].weight.add_(
            (self.config.dt / self.config.tau_weights) * self.dwpf[-1]
        )

    def updateHist(self, hist, tab, param=False):
        # instantiate a first empty tensor that is on the same device as the net
        if hist == []:
            hist = [torch.empty(0, device=torch.device("cpu")) for _ in range(len(tab))]

        if not param:
            for k in range(len(tab)):
                hist[k] = torch.cat(
                    (hist[k], tab[k][0].clone().unsqueeze(1).cpu()), dim=1
                )
        else:
            for k in range(len(tab)):
                hist[k] = torch.cat(
                    (hist[k], tab[k].weight.clone().unsqueeze(2).cpu()), dim=2
                )
        return hist

    def save_weights(self, filename, dir=None):
        if dir is None:
            now = datetime.datetime.now()
            date_string = now.strftime("%Y-%m-%d")
            # make sure the directory graphics/date_string exists
            os.makedirs(r"weights/" + date_string, exist_ok=True)
            dir = r"weights/" + date_string
        else:
            os.makedirs(dir, exist_ok=True)

        # Save state_dicts of the ModuleLists
        torch.save(
            {
                "wpf": self.wpf.state_dict(),
                "wpb": self.wpb.state_dict(),
                "wip": self.wip.state_dict(),
                "wpi": self.wpi.state_dict(),
            },
            dir + "/" + filename,
        )

    def load_weights(self, file_path):
        # Load state_dicts into the ModuleLists
        checkpoint = torch.load(file_path, weights_only=True)
        self.wpf.load_state_dict(checkpoint["wpf"])
        for w in self.wpf:
            assert not torch.isnan(w.weight).any(), "NaN found in wpf after loading"
        self.wpb.load_state_dict(checkpoint["wpb"])
        for w in self.wpb:
            assert not torch.isnan(w.weight).any(), "NaN found in wpb after loading"
        self.wip.load_state_dict(checkpoint["wip"])
        for w in self.wip:
            assert not torch.isnan(w.weight).any(), "NaN found in wip after loading"
        self.wpi.load_state_dict(checkpoint["wpi"])
        for w in self.wpi:
            assert not torch.isnan(w.weight).any(), "NaN found in wpi after loading"

    def extra_repr(self):
        print(self.config)


class teacherNet(nn.Module):
    def __init__(self, size_tab, k_tab):
        super(teacherNet, self).__init__()

        self.topology = size_tab

        self.gamma = 0.1
        self.beta = 1
        self.theta = 3
        self.k_tab = k_tab

        w = nn.ModuleList([])

        # Build weights
        for i in range(len(self.topology) - 1):
            w.append(nn.Linear(self.topology[i], self.topology[i + 1], bias=False))
            torch.nn.init.uniform_(w[i].weight, a=-1, b=1)

        self.w = w

    def rho(self, x):
        return self.gamma * torch.log(1 + torch.exp(self.beta * (x - self.theta)))

    def forward(self, x):
        a = x
        for i in range(len(self.w)):
            a = self.rho(self.k_tab[i] * self.w[i](a))

        return a
