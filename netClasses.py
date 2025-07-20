from __future__ import print_function

import datetime
import os

import numpy as np
import torch
import torch.nn as nn


class DendriticNet(nn.Module):
    def __init__(self, size_tab, initw, dt, gsom, glk, gb, ga, gd, tau_weights, noise, rho, lr_pf, lr_ip, lr_pi, lr_pb):
        super(DendriticNet, self).__init__()
        self.device = torch.device("cpu")
        self.topology = size_tab
        self.layers = len(size_tab) - 1
        self.dt = dt 
        self.gsom = gsom
        self.glk = glk
        self.gb = gb
        self.ga = ga
        self.gd = gd
        self.tau_weights = tau_weights
        self.noise = noise
        self.rho = rho
        self.lr_pf = lr_pf
        self.lr_ip = lr_ip
        self.lr_pi = lr_pi
        self.lr_pb = lr_pb
        # Initialize state vectors
        self.initialize_states()
        self.initialize_weights(initw)   

    def initialize_weights(self, initw: float):
        self.wpf = nn.ModuleList([])
        self.wpb = nn.ModuleList([])
        self.wpi = nn.ModuleList([])
        self.wip = nn.ModuleList([])

        # Build input weights
        self.wpf.append(nn.Linear(self.topology[0], self.topology[1], bias=False))

        # Build weights for hidden layers
        for hidden_layer in range(1, self.layers):
            self.wpf.append(
                nn.Linear(self.topology[hidden_layer], self.topology[hidden_layer + 1], bias=False)
            )
            torch.nn.init.uniform_(
                self.wpf[-1].weight, a=-initw, b=initw
            )
            self.wpb.append(
                nn.Linear(self.topology[hidden_layer + 1], self.topology[hidden_layer], bias=False)
            )
            torch.nn.init.uniform_(
                self.wpb[-1].weight, a=-initw, b=initw
            )
            self.wip.append(
                nn.Linear(self.topology[hidden_layer], self.topology[hidden_layer + 1], bias=False)
            )
            torch.nn.init.uniform_(
                self.wip[-1].weight, a=-initw, b=initw
            )
            self.wpi.append(
                nn.Linear(self.topology[hidden_layer + 1], self.topology[hidden_layer], bias=False)
            )
            torch.nn.init.uniform_(
                self.wpi[-1].weight, a=-initw, b=initw
            )

        # Initialize weight derivatives
        self.dwpf = [torch.zeros_like(w.weight, device=self.device) for w in self.wpf]
        self.dwpi = [torch.zeros_like(w.weight, device=self.device) for w in self.wpi]
        self.dwpb = [torch.zeros_like(w.weight, device=self.device) for w in self.wpb]
        self.dwip = [torch.zeros_like(w.weight, device=self.device) for w in self.wip]

    def initialize_states(self, batch_size: int=1):
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
        g_exc = self.gsom * (u_trgt - e_inh) / (e_exc - e_inh)
        # g^P_{inh,N} = -g_{som} \frac{u^{trgt}_{N} - E_{exc}}{E_{exc} - E_{inh}}
        g_inh = -self.gsom * (u_trgt - e_exc) / (e_exc - e_inh)
        # i^P_{N}(t) = g^P_{exc,N} (E_{exc} - u^P_{N}(t)) + g^P_{inh,N} (E_{inh} - u^P_{N}(t))
        i_nudge = g_exc * (e_exc - u) + g_inh * (e_inh - u)
        return i_nudge

    def stepper(self, data, target=None, plasticity=False):
        """
        Update neuron states and synapses based on input data and compute apical potentials.

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

        # Loop over hiden_layers
        for hidden_layer in range(self.layers - 1):
            input_rho = self.rho(data) if hidden_layer == 0 else self.r_p[hidden_layer - 1]
            # v^P_{B,k}(t) = W^{PP}_{k,k-1} \phi(u^P_{k-1}(t)) (3)
            vb = self.wpf[hidden_layer](input_rho)
            # v^I_{k}(t) = W^{IP}_{k,k} \phi(u^P_{k}(t)) (5)
            vi = self.wip[hidden_layer](self.r_p[hidden_layer])
            # v^P_{A,k}(t) = W^{PP}_{k,k+1} \phi(u^P_{k+1}(t))+ W^{PI}_{k,k} \phi(u^I_{k}(t)) (4)
            va = self.wpb[hidden_layer](self.r_p[hidden_layer + 1]) + self.wpi[hidden_layer](self.r_i[hidden_layer])

            va_topdown.append(self.wpb[hidden_layer](self.r_p[hidden_layer + 1]))
            va_cancelation.append(self.wpi[hidden_layer](self.r_i[hidden_layer]))
            vb_input.append(vb)

            # interneuron nudging calculation
            # i_nudge = self.compute_nudging(self.u_i[k], self.u_p[k + 1])
            i_nudge = self.gsom * (self.u_p[hidden_layer + 1] - self.u_i[hidden_layer])

            # Update derivatives
            # \frac{d}{dt}u^P_{k}(t) = -g_{lk}u^P_{k}(t) + g_{b}(v^P_{B,k}(t) - u^P_{k}(t)) + g_{a}(v^P_{A,k}(t) - u^P_{k}(t)) + \sigma\xi(t) (1)
            self.du_p[hidden_layer] = (
                -self.glk * self.u_p[hidden_layer]
                + self.gb * (vb - self.u_p[hidden_layer])
                + self.ga * (va - self.u_p[hidden_layer])
                + self.noise * torch.randn_like(self.u_p[hidden_layer], device=self.device)
            )

            # \frac{d}{dt}u^I_{k}(t) = -g_{lk}u^I_{k}(t) + g_{d}(v^I_{k}(t) - u^I_{k}(t)) + i^I_{k}(t) + \sigma\xi(t) (2)
            self.du_i[hidden_layer] = (
                -self.glk * self.u_i[hidden_layer]
                + self.gd * (vi - self.u_i[hidden_layer])
                + i_nudge
                + self.noise * torch.randn_like(self.u_i[hidden_layer], device=self.device)
            )

        # Compute derivative of the output layer
        # v^P_{B,k}(t) = W^{PP}_{k,k-1} \phi(u^P_{k-1}(t)) (3)
        vb = self.wpf[-1](self.r_p[-2])
        vb_input.append(vb)
        # nudge output towards target
        if target is not None:
            # i_nudge = self.compute_nudging(self.u_p[-1], target)
            i_nudge = self.gsom * (target - self.u_p[-1])
        else:
            i_nudge = torch.zeros_like(self.u_p[-1])

        # \frac{d}{dt}u^P_{k}(t) = -g_{lk}u^P_{k}(t) + g_{b}(v^P_{B,k}(t) - u^P_{k}(t)) + i^P_{N}(t) + \sigma\xi(t)
        self.du_p[-1].copy_(
            -self.glk * self.u_p[-1]
            + self.gb * (vb - self.u_p[-1])
            + i_nudge
            + self.noise * torch.randn_like(self.u_p[-1], device=self.device)
        )

        # Update the values of the neurons in the hidden layers by adding the time step times the derivative
        for hidden_layer in range(self.hidden_layer - 1):
            self.u_p[hidden_layer].add_(self.dt * self.du_p[hidden_layer])
            self.u_i[hidden_layer].add_(self.dt * self.du_i[hidden_layer])

        self.u_p[-1].add_(self.dt * self.du_p[-1])

        # update the rates
        self.r_p = [self.rho(u_p) for u_p in self.u_p]
        self.r_i = [self.rho(u_i) for u_i in self.u_i]

        return [va_topdown, va_cancelation, vb_input]

    def updateWeights(self, data):
        for hidden_layer in range(self.hidden_layer - 1):
            # v^P_{B,k}(t) = W^{PP}_{k,k-1} \phi(u^P_{k-1}(t)) (3)
            vb = self.wpf[hidden_layer](self.rho(data) if hidden_layer == 0 else self.r_p[hidden_layer - 1])
            # v^I_{k}(t) = W^{IP}_{k,k} \phi(u^P_{k}(t)) (5)
            vi = self.wip[hidden_layer](self.r_p[hidden_layer])
            # v^P_{A,k}(t) = W^{PP}_{k,k+1} \phi(u^P_{k+1}(t))+ W^{PI}_{k,k} \phi(u^I_{k}(t)) (4)
            va = self.wpi[hidden_layer](self.r_i[hidden_layer]) + self.wpb[hidden_layer](self.r_p[hidden_layer + 1])

            # \hat{v}^P_{TD,k} = W_{k,k+1} r_p_{k+1}
            vtd = self.wpb[hidden_layer](self.r_p[hidden_layer + 1])

            # voltage of the basal compartment when taking into account the dendritic attenuation factors
            # \hat{v}^P_{B,k} = \frac{g_b}{g_{lk} + g_b + g_a} v^P_{B,k}
            vbhat = (
                self.gb
                / (self.glk + self.gb + self.ga)
                * vb
            )
            # \hat{v}^I_k = \frac{g_d}{g_{lk} + g_d} v^I_k
            vihat = self.gd / (self.gd + self.glk) * vi

            # \frac{d}{dt}W^{PP}_{k,k-1} = \nabla^{PP}_{k,k-1}(\phi(u^P_k)-\phi(\hat{v}^P_{B,k}))(r^P_{k-1})^T (7)
            self.dwpf[hidden_layer].copy_(
                self.lr_pf[hidden_layer]
                * torch.mm(
                    torch.transpose(self.r_p[hidden_layer] - self.rho(vbhat), 0, 1),
                    (self.rho(data)) if hidden_layer == 0 else self.r_p[hidden_layer - 1],
                )
            )
            # \frac{d}{dt}W^{IP}_{k,k} = \nabla^{IP}_{k,k}(\phi(u^I_k)-\phi(\hat{v}^I_k))(r^I_k)^T (8)
            self.dwip[hidden_layer].copy_(
                self.lr_ip[hidden_layer]
                * torch.mm(
                    torch.transpose(self.r_i[hidden_layer] - self.rho(vihat), 0, 1),
                    self.r_p[hidden_layer],
                )
            )
            # \frac{d}{dt}W^{PI}_{k,k} = \nabla^{PI}_{k,k}(v_{rest}-v^P_{A,k}))(r^I_k)^T (9)
            self.dwpi[hidden_layer].copy_(
                self.lr_pi[hidden_layer] * torch.mm(torch.transpose(-va, 0, 1), self.r_i[hidden_layer])
            )
            # \frac{d}{dt}W^{PP}_{k,k+1} = \nabla^{PP}_{k,k+1}(\phi(u^P_k)-\phi(\hat{v}^P_{TD,k}))(r^P_{k+1})^T (10)
            self.dwpb[hidden_layer].copy_(
                self.lr_pb[hidden_layer]
                * torch.mm(
                    torch.transpose(self.r_p[hidden_layer] - self.rho(vtd), 0, 1),
                    self.r_p[hidden_layer + 1],
                )
            )

        # for the output layer
        # v^P_{B,N}(t) = W^{PP}_{k,k-1} \phi(u^P_{k-1}(t)) (3)
        vb = self.wpf[-1](self.r_p[-2])
        # \hat{v}^P_{B,k} = \frac{g_b}{g_{lk} + g_b + g_a} v^P_{B,k}
        vbhat = (
            self.gb / (self.glk + self.gb + self.ga) * vb
        )
        # \frac{d}{dt}W^{PP}_{k,k-1} = \nabla^{PP}_{k,k-1}(\phi(u^P_k)-\phi(\hat{v}^P_{B,k}))(r^P_{k-1})^T (7)
        self.dwpf[-1].copy_(
            self.config.lr_pf[-1]
            * torch.mm(
                torch.transpose(self.r_p[-1] - self.rho(vbhat), 0, 1),
                self.r_p[-2],
            )
        )
        for hidden_layer in range(self.layers - 1):
            self.wpf[hidden_layer].weight.add_(
                (self.dt / self.tau_weights) * self.dwpf[hidden_layer]
            )
            self.wpb[hidden_layer].weight.add_(
                (self.dt / self.tau_weights) * self.dwpb[hidden_layer]
            )
            self.wpi[hidden_layer].weight.add_(
                (self.dt / self.tau_weights) * self.dwpi[hidden_layer] 
            )
            self.wip[hidden_layer].weight.add_(
                (self.dt / self.tau_weights) * self.dwip[hidden_layer]
            )
        self.wpf[-1].weight.add_(
            (self.dt / self.tau_weights) * self.dwpf[-1]
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

        # Save weight tensors from ModuleLists
        torch.save(
            {
                "wpf": [w.state_dict() for w in self.wpf],
                "wpb": [w.state_dict() for w in self.wpb],
                "wip": [w.state_dict() for w in self.wip],
                "wpi": [w.state_dict() for w in self.wpi],
            },
            dir + "/" + filename,
        )

    def load_weights(self, file_path):
        # Load state_dicts into the ModuleLists
        checkpoint = torch.load(file_path, weights_only=True)
        for i, state in enumerate(checkpoint["wpf"]):
            self.wpf[i].load_state_dict(state)
            assert not torch.isnan(self.wpf[i].weight).any(), "NaN found in wpf after loading"
        for i, state in enumerate(checkpoint["wpb"]):
            self.wpb[i].load_state_dict(state)
            assert not torch.isnan(self.wpb[i].weight).any(), "NaN found in wpb after loading"
        for i, state in enumerate(checkpoint["wip"]):
            self.wip[i].load_state_dict(state)
            assert not torch.isnan(self.wip[i].weight).any(), "NaN found in wip after loading"
        for i, state in enumerate(checkpoint["wpi"]):
            self.wpi[i].load_state_dict(state)
            assert not torch.isnan(self.wpi[i].weight).any(), "NaN found in wpi after loading"

    def extra_repr(self):
        # return a string representation of the class
        return (
            f"topology={self.topology}, "
            f"depth={self.depth}, "
            f"dt={self.dt}, "
            f"gsom={self.gsom}, "
            f"glk={self.glk}, "
            f"gb={self.gb}, "
            f"ga={self.ga}, "
            f"gd={self.gd}, "
            f"tau_weights={self.tau_weights}, "
            f"noise={self.noise}",
            f"rho={self.rho.__name__}, "
            f"lr_pf={self.lr_pf}, "
            f"lr_ip={self.lr_ip}, "
            f"lr_pi={self.lr_pi}, "
            f"lr_pb={self.lr_pb}"
        )
    
    def __to__(self, device: torch.device):
        self.device = device
        self.u_p = [u_p.to(device) for u_p in self.u_p]
        self.u_i = [u_i.to(device) for u_i in self.u_i]
        self.r_p = [r_p.to(device) for r_p in self.r_p]
        self.r_i = [r_i.to(device) for r_i in self.r_i]
        self.du_p = [du_p.to(device) for du_p in self.du_p]
        self.du_i = [du_i.to(device) for du_i in self.du_i]
        self.wpf = [w.weight.to(device) for w in self.wpf]
        self.wpb = [w.weight.to(device) for w in self.wpb]
        self.wip = [w.weight.to(device) for w in self.wip]
        self.wpi = [w.weight.to(device) for w in self.wpi]
        self.dwip = [dwip.to(device) for dwip in self.dwip]
        self.dwpi = [dwpi.to(device) for dwpi in self.dwpi]
        self.dwip = [dwip.to(device) for dwip in self.dwip]
        self.dwpi = [dwpi.to(device) for dwpi in self.dwpi]


class TeacherNet(nn.Module):
    def __init__(self, size_tab, k_tab):
        super(TeacherNet, self).__init__()

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
