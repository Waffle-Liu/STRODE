import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint as odeint
import torch.nn.functional as F
from sru import SRU

from config import opt
from utils import *

class NonnegativeLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # Make weight non-negative at initialization
        self.weight.data.abs_()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        self.weight.data.clamp_(0.0)
        return F.linear(input, self.weight, self.bias)


class NonpositiveLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # Make weight non-positive at initialization
        self.weight.data.abs_()
        self.weight.data.mul_(-1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        self.weight.data.clamp_(max=0.0)
        return F.linear(input, self.weight, self.bias)


class FullNN_Phi(nn.Module):
    def __init__(self, h_dim, time_dim=64, n_full_layers=2):
        super(FullNN_Phi, self).__init__()
        self.time_dim = time_dim
        self.n_full_layers = n_full_layers

        self.linear_time = NonpositiveLinear(1, time_dim, bias=False)
        self.linear_time_layers = nn.ModuleList(
            [NonpositiveLinear(time_dim, time_dim, bias=False) for _ in range(n_full_layers - 1)])
        self.final_time_layer = NonpositiveLinear(time_dim, 1, bias=False)

        self.dense_p = NonpositiveLinear(h_dim, time_dim, bias=False)
        self.linear_p_layers = nn.ModuleList(
            [NonpositiveLinear(time_dim, time_dim, bias=False) for _ in range(n_full_layers - 1)])
        self.final_p_layer = NonpositiveLinear(time_dim, 1, bias=False)
        self.c = 2.0

    def mlp(self, input_x, input_rnn):
        hidden_x = self.linear_time(input_x)
        for i in range(self.n_full_layers-1):
            hidden_x = self.linear_time_layers[i](hidden_x)
            hidden_x = torch.tanh(hidden_x)
        hidden_x = self.final_time_layer(hidden_x)

        hidden_p = self.dense_p(input_rnn)
        for i in range(self.n_full_layers-1):
            hidden_p = self.linear_p_layers[i](hidden_p)
            hidden_p = torch.tanh(hidden_p)
        hidden_p = self.final_p_layer(hidden_p)

        hidden = hidden_x+hidden_p
        hidden = F.softplus(hidden)

        return hidden + input_x.detach()

    def forward(self, input_x, input_rnn, mode='train'):

        integral = self.mlp(input_x, input_rnn)

        if mode == 'train':
            derivative = torch.autograd.grad(
                integral, input_x, torch.ones_like(integral), create_graph=True)[0]
        else:
            derivative = torch.ones_like(integral)

        return integral, derivative


class FullNN_phi(nn.Module):
    def __init__(self, h_dim, time_dim=64, n_full_layers=2):
        super(FullNN_phi, self).__init__()
        self.time_dim = time_dim
        self.n_full_layers = n_full_layers

        self.linear_time = NonnegativeLinear(1, time_dim)
        self.linear_time_layers = nn.ModuleList(
            [NonnegativeLinear(time_dim, time_dim) for _ in range(n_full_layers - 1)])
        self.final_time_layer = NonnegativeLinear(time_dim, 1)

        self.dense_p = NonnegativeLinear(h_dim, time_dim)
        self.linear_p_layers = nn.ModuleList(
            [NonnegativeLinear(time_dim, time_dim) for _ in range(n_full_layers - 1)])
        self.final_p_layer = NonnegativeLinear(time_dim, 1)

    def mlp(self, input_x, input_rnn):
        hidden_x = self.linear_time(input_x)
        for i in range(self.n_full_layers-1):
            hidden_x = self.linear_time_layers[i](hidden_x)
            hidden_x = torch.tanh(hidden_x)
        hidden_x = self.final_time_layer(hidden_x)

        hidden_p = self.dense_p(input_rnn)
        for i in range(self.n_full_layers-1):
            hidden_p = self.linear_p_layers[i](hidden_p)
            hidden_p = torch.tanh(hidden_p)
        hidden_p = self.final_p_layer(hidden_p)

        hidden = hidden_x+hidden_p
        hidden = F.softplus(hidden)

        return hidden + input_x.detach()

    def forward(self, input_x, input_rnn, mode='train'):

        integral = self.mlp(input_x, input_rnn)

        if mode == 'train':
            derivative = torch.autograd.grad(
                integral, input_x, torch.ones_like(integral), create_graph=True)[0]
        else:
            derivative = torch.ones_like(integral)

        return integral, derivative


def create_net(n_inputs, n_outputs, n_layers=1,
               n_units=100, nonlinear=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)

# Adapted from LatentODE(https://github.com/YuliaRubanova/latent_ode)
class ODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, device=torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device

        init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards=False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)


class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, latents,
                 odeint_rtol=1e-4, odeint_atol=1e-5, device=torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_func = ode_func
        self.ode_method = method
        self.latents = latents
        self.device = device
        
    def forward(self, y0, t_seq):
        k = opt.ode1_step
        delta_t = t_seq[..., 1].unsqueeze(-1) - t_seq[..., 0].unsqueeze(-1)
        delta_t = delta_t/k
        ti = t_seq[..., 0].unsqueeze(-1)
        yi = y0

        for i in range(k):
            yi = self.perform_one_step(self.ode_func, yi, ti, delta_t)
            ti = ti + delta_t

        return yi

    def perform_one_step(self, func, y0, t, dt):
        y = func(t, y0)
        dy = y*dt
        return y0 + dy


class DiffeqSolver_KL(nn.Module):
    def __init__(self, device, c=1.000001, y0=0.):
        super(DiffeqSolver_KL, self).__init__()
        self.device = device
        self.epsilons = opt.ode2_eps
        self.c = torch.tensor(c).to(device)
        self.y0 = torch.tensor(y0).to(device)

    def forward(self, b_phi_zt, b_phi_zt_deriv, s_phi_zt, s_phi_zt_deriv):
        end_m = 1.0/self.epsilons
        solutions = []
        y_pre = self.y0
        const = 0.01
        K = s_phi_zt + torch.log(-1.*b_phi_zt_deriv+const) - \
            torch.log(s_phi_zt_deriv+const)
        eps = torch.tensor(self.epsilons).to(self.device)
        y = torch.zeros_like(s_phi_zt).to(self.device)

        def _gm(m):
            return -1.*b_phi_zt_deriv/(m*torch.log(-m)) * (K - torch.log(-torch.log(-m)))

        for t in range(1, int(end_m)):
            m = torch.tensor(-1 + self.epsilons*t).to(self.device)
            dt = eps
            gm = _gm(m)
            dy = gm*dt
            y = y_pre + dy
            solutions.append(y)
            y_pre = y

        first_item = solutions[-1]
        # Gm_eps = torch.abs(solutions[-2]-solutions[-1])
        Gm_eps = solutions[-2]-solutions[-1]
        second_item = Gm_eps

        loss = first_item+second_item

        return loss

#########################################################################
######################## Implementation of Model ########################
#########################################################################

class NODE(nn.Module):
    def __init__(self, in_dim, h_dim=128, kernel_size=5, device=torch.device('cuda')):
        super(NODE, self).__init__()
        
        ode_in_dim = 4*h_dim
        ode_latent_dim = ode_in_dim
        self.device = device

        self.ndf = h_dim
        self.encode_xt = nn.Sequential(
            # 1*64*64
            nn.Conv2d(in_dim, self.ndf, kernel_size, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128*16*16
            nn.Conv2d(self.ndf, 2*self.ndf, kernel_size, 2, 2, bias=False),
            nn.BatchNorm2d(2*self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 256*8*8
            nn.Conv2d(2*self.ndf, 4*self.ndf, kernel_size, 2, 2, bias=False),
            nn.BatchNorm2d(4*self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 512*4*4
            nn.Conv2d(4*self.ndf, 4*self.ndf, 4, 1, 0, bias=False)
            # 128*1*1
        )        

        self.ode_func_net = create_net(ode_in_dim, ode_latent_dim)
        self.odefunc = ODEFunc(input_dim=ode_in_dim, latent_dim=ode_latent_dim, ode_func_net=self.ode_func_net)
        self.ode_solver = DiffeqSolver(input_dim=ode_in_dim, ode_func=self.odefunc, method='euler', latents=10, device=device)
        
        self.deconv = nn.Sequential(
            # 512*1*1
            nn.ConvTranspose2d(4*self.ndf, 4*self.ndf, 4, 4, 0, bias=False),
            nn.BatchNorm2d(4*self.ndf),
            nn.ReLU(inplace=True),
            # 512*4*4
            nn.ConvTranspose2d(4*self.ndf, 2*self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*self.ndf),
            nn.ReLU(inplace=True),
            # 256*8*8
            nn.ConvTranspose2d(2*self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.ReLU(inplace=True),
            # 128*16*16
            nn.ConvTranspose2d(self.ndf, in_dim, 4, 4, 0, bias=False),
            nn.Tanh()
            # 1*64*64
        )

    def forward(self, x, mode='train'):
        b, l ,c, w, h = x.shape
        x = x.reshape(b*l, c, w, h)
        zt_post = self.encode_xt(x)
        zt_post = zt_post.reshape(b, l, -1)

        integral1_list = torch.arange(0, l).repeat((b, 1)).unsqueeze(-1).to(self.device)
        integral1_after_list = torch.cat((integral1_list[:, 1:, :], integral1_list[:, -1:, :]), dim=1)

        ode_before = zt_post
        zt_inv_list = []
        for j in range(integral1_list.shape[1]):
            # perform one step
            t_now = integral1_list[:, j, :]
            t_after = integral1_after_list[:, j, :]
            t_seq = torch.cat((t_now, t_after), dim=-1)
            zt_inv = self.ode_solver(zt_post[:, j, :], t_seq)
            zt_inv_list.append(zt_inv)
        qzt_inv = torch.stack(zt_inv_list, dim=1)
        ode_after = qzt_inv

        qzt_inv = qzt_inv.reshape(b*l, 4*self.ndf, 1, 1)
        qzt_deconv = self.deconv(qzt_inv)
        qzt_deconv = qzt_deconv.reshape(b, l, c, w, h) 

        x = x.reshape(b, l, c, w, h)
        x_next = torch.cat((x[:, 1:, :, :, :], x[:, -1:, :, :, :]), dim=1)
        mse = torch.mean(F.mse_loss(qzt_deconv, x_next))
        loss = mse 

        
        return loss, mse, 0, integral1_list.reshape(b, l)

class NODE_RNN(nn.Module):
    def __init__(self, in_dim, h_dim=128, kernel_size=5, device=torch.device('cuda')):
        super(NODE_RNN, self).__init__()
       
        ode_in_dim = 4*h_dim
        ode_latent_dim = ode_in_dim
        self.device = device

        self.ndf = h_dim
        self.encode_xt = nn.Sequential(
            # 1*64*64
            nn.Conv2d(in_dim, self.ndf, kernel_size, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128*16*16
            nn.Conv2d(self.ndf, 2*self.ndf, kernel_size, 2, 2, bias=False),
            nn.BatchNorm2d(2*self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 256*8*8
            nn.Conv2d(2*self.ndf, 4*self.ndf, kernel_size, 2, 2, bias=False),
            nn.BatchNorm2d(4*self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 512*4*4
            nn.Conv2d(4*self.ndf, 4*self.ndf, 4, 1, 0, bias=False)
            # 128*1*1
        )        

        self.ode_func_net = create_net(ode_in_dim, ode_latent_dim)
        self.odefunc = ODEFunc(input_dim=ode_in_dim, latent_dim=ode_latent_dim, ode_func_net=self.ode_func_net)
        self.ode_solver = DiffeqSolver(input_dim=ode_in_dim, ode_func=self.odefunc, method='euler', latents=10, device=device)
        self.rnn = SRU(input_size=ode_latent_dim, hidden_size=ode_latent_dim, num_layers=2)

        self.deconv = nn.Sequential(
            # 512*1*1
            nn.ConvTranspose2d(4*self.ndf, 4*self.ndf, 4, 4, 0, bias=False),
            nn.BatchNorm2d(4*self.ndf),
            nn.ReLU(inplace=True),
            # 512*4*4
            nn.ConvTranspose2d(4*self.ndf, 2*self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*self.ndf),
            nn.ReLU(inplace=True),
            # 256*8*8
            nn.ConvTranspose2d(2*self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.ReLU(inplace=True),
            # 128*16*16
            nn.ConvTranspose2d(self.ndf, in_dim, 4, 4, 0, bias=False),
            nn.Tanh()
            # 1*64*64
        )

    def forward(self, x, mode='train'):
        b, l ,c, w, h = x.shape
        x = x.reshape(b*l, c, w, h)
        zt_post = self.encode_xt(x)
        zt_post = zt_post.reshape(b, l, -1)

        integral1_list = torch.arange(0, l).repeat((b, 1)).unsqueeze(-1).to(self.device)
        integral1_after_list = torch.cat((integral1_list[:, 1:, :], integral1_list[:, -1:, :]), dim=1)
        # post q_zt
        ode_before = zt_post
        zt_inv_list = []
        for j in range(integral1_list.shape[1]):
            # perform one step
            t_now = integral1_list[:, j, :]
            t_after = integral1_after_list[:, j, :]
            t_seq = torch.cat((t_now, t_after), dim=-1)
            zt_inv = self.ode_solver(zt_post[:, j, :], t_seq)
            zt_inv_list.append(zt_inv)
        qzt_inv = torch.stack(zt_inv_list, dim=1)
        input_rnn = qzt_inv.permute(1, 0, 2)
        h0 = torch.zeros((2, b, 4*self.ndf)).to(self.device)
        ht, _ = self.rnn(input_rnn, h0)
        ht = ht.permute(1, 0, 2)
        ode_after = qzt_inv
        
        qzt_inv = ht
        qzt_inv = qzt_inv.reshape(b*l, 4*self.ndf, 1, 1)
        qzt_deconv = self.deconv(qzt_inv)
        qzt_deconv = qzt_deconv.reshape(b, l, c, w, h) 

        # reconstruction loss
        x = x.reshape(b, l, c, w, h)
        x_next = torch.cat((x[:, 1:, :, :, :], x[:, -1:, :, :, :]), dim=1)
        
        mse = torch.mean(F.mse_loss(qzt_deconv, x_next))
        loss = mse 

        # return loss, decode_out, likelihood, kl2, ode_before, ode_after
        return loss, mse, 0, integral1_list.reshape(b, l)

class STRODE(nn.Module):
    def __init__(self, in_dim, h_dim=128, kernel_size=5, time_dim1=64, n_full_layers1=2, time_dim2=64, n_full_layers2=2, device=torch.device('cuda')):
        super(STRODE, self).__init__()
        # encoder for big phi
        ode_in_dim = 4*h_dim
        ode_latent_dim = ode_in_dim
        self.device = device

        self.ndf = h_dim
        self.encode_xt = nn.Sequential(
            # 1*64*64
            nn.Conv2d(in_dim, self.ndf, kernel_size, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128*16*16
            nn.Conv2d(self.ndf, 2*self.ndf, kernel_size, 2, 2, bias=False),
            nn.BatchNorm2d(2*self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 256*8*8
            nn.Conv2d(2*self.ndf, 4*self.ndf, kernel_size, 2, 2, bias=False),
            nn.BatchNorm2d(4*self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 512*4*4
            nn.Conv2d(4*self.ndf, 4*self.ndf, 4, 1, 0, bias=False)
            # 128*1*1
        )
        self.fullnn1 = FullNN_Phi(4*h_dim, time_dim1, n_full_layers1)
        
        self.encode_prior = nn.Sequential(
            # 1*64*64
            nn.Conv2d(in_dim, self.ndf, kernel_size, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128*16*16
            nn.Conv2d(self.ndf, 2*self.ndf, kernel_size, 2, 2, bias=False),
            nn.BatchNorm2d(2*self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 256*8*8
            nn.Conv2d(2*self.ndf, 4*self.ndf, kernel_size, 2, 2, bias=False),
            nn.BatchNorm2d(4*self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 512*4*4
            nn.Conv2d(4*self.ndf, 4*self.ndf, 4, 1, 0, bias=False)
            # 512*1*1
        )
        self.fullnn2 = FullNN_phi(4*h_dim, time_dim2, n_full_layers2)

        self.ode_func_net = create_net(ode_in_dim, ode_latent_dim)
        self.odefunc = ODEFunc(input_dim=ode_in_dim, latent_dim=ode_latent_dim, ode_func_net=self.ode_func_net)
        self.ode_solver = DiffeqSolver(input_dim=ode_in_dim, ode_func=self.odefunc, method='euler', latents=10, device=device)
        self.ode_solver_loss = DiffeqSolver_KL(device=device)
       
        self.deconv = nn.Sequential(
            # 512*1*1
            nn.ConvTranspose2d(4*self.ndf, 4*self.ndf, 4, 4, 0, bias=False),
            nn.BatchNorm2d(4*self.ndf),
            nn.ReLU(inplace=True),
            # 512*4*4
            nn.ConvTranspose2d(4*self.ndf, 2*self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*self.ndf),
            nn.ReLU(inplace=True),
            # 256*8*8
            nn.ConvTranspose2d(2*self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.ReLU(inplace=True),
            # 128*16*16
            nn.ConvTranspose2d(self.ndf, in_dim, 4, 4, 0, bias=False),
            nn.Tanh()
            # 1*64*64
        )

    def forward(self, x, mode='train'):
        b, l ,c, w, h = x.shape
        x = x.reshape(b*l, c, w, h)
        zt_post = self.encode_xt(x)
        zt_post = zt_post.reshape(b, l, -1)
        input_x = torch.zeros((b, 1), requires_grad=True).to(self.device)
        integral1_list = []
        derivative1_list = []
        for i in range(l):
            integral1, derivative1 = self.fullnn1(
                input_x, zt_post[:, i, :], mode)
            integral1_list.append(integral1)
            derivative1_list.append(derivative1)
            input_x = integral1
        integral1_list = torch.stack(integral1_list, dim=1)
        derivative1_list = torch.stack(derivative1_list, dim=1)
        
        integral1_after_list = torch.cat((integral1_list[:, 1:, :], integral1_list[:, -1:, :]), dim=1)
      
        zt_prior = self.encode_prior(x)
        zt_prior = zt_prior.reshape(b, l, -1)
        input_x = torch.zeros((b, 1), requires_grad=True).to(self.device)
        integral2_list = []
        derivative2_list = []
        for i in range(l):
            integral2, derivative2 = self.fullnn2(
                input_x, zt_prior[:, i, :], mode)
            integral2_list.append(integral2)
            derivative2_list.append(derivative2)
            input_x = integral2
        integral2_list = torch.stack(integral2_list, dim=1)
        derivative2_list = torch.stack(derivative2_list, dim=1)

        # post q_zt
        ode_before = zt_post
        zt_inv_list = []
        for j in range(integral1_list.shape[1]):
            t_now = integral1_list[:, j, :]
            t_after = integral1_after_list[:, j, :]
            t_seq = torch.cat((t_now, t_after), dim=-1)
            zt_inv = self.ode_solver(zt_post[:, j, :], t_seq)
            zt_inv_list.append(zt_inv)
        qzt_inv = torch.stack(zt_inv_list, dim=1)
        ode_after = qzt_inv

        qzt_inv = qzt_inv.reshape(b*l, 4*self.ndf, 1, 1)
        qzt_deconv = self.deconv(qzt_inv)
        qzt_deconv = qzt_deconv.reshape(b, l, c, w, h) 

        kl2 = self.kl_pt_qt(integral1_list, derivative1_list,
                            integral2_list, derivative2_list)

        # reconstruction loss
        x = x.reshape(b, l, c, w, h)
        x_next = torch.cat((x[:, 1:, :, :, :], x[:, -1:, :, :, :]), dim=1)
        mse = torch.mean(F.mse_loss(qzt_deconv, x_next))

        loss = mse + opt.w_kl*kl2

        return loss, mse, kl2, integral1_list.reshape(b, l)

    def kl_pt_qt(self, integral1, derivative1, integral2, derivative2):
        loss = torch.zeros((opt.train_batch_size, opt.video_length)).to(self.device)
        for i in range(opt.video_length):
            loss_i = self.ode_solver_loss(integral1[:, i, :], derivative1[:, i, :], integral2[:, i, :], derivative2[:, i, :])
            loss[:, i:i+1] = loss_i

        return torch.mean(loss).to(self.device)

class STRODE_RNN(nn.Module):
    def __init__(self, in_dim, h_dim=128, kernel_size=5, time_dim1=64, n_full_layers1=2, time_dim2=64, n_full_layers2=2, device=torch.device('cuda')):
        super(STRODE_RNN, self).__init__()
        ode_in_dim = 4*h_dim
        ode_latent_dim = ode_in_dim
        self.device = device
        
        self.ndf = h_dim
        self.encode_xt = nn.Sequential(
            # 1*64*64
            nn.Conv2d(in_dim, self.ndf, kernel_size, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128*16*16
            nn.Conv2d(self.ndf, 2*self.ndf, kernel_size, 2, 2, bias=False),
            nn.BatchNorm2d(2*self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 256*8*8
            nn.Conv2d(2*self.ndf, 4*self.ndf, kernel_size, 2, 2, bias=False),
            nn.BatchNorm2d(4*self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 512*4*4
            nn.Conv2d(4*self.ndf, 4*self.ndf, 4, 1, 0, bias=False)
            # 128*1*1
        )
        self.fullnn1 = FullNN_Phi(4*h_dim, time_dim1, n_full_layers1)
        
        self.encode_prior = nn.Sequential(
            # 1*64*64
            nn.Conv2d(in_dim, self.ndf, kernel_size, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128*16*16
            nn.Conv2d(self.ndf, 2*self.ndf, kernel_size, 2, 2, bias=False),
            nn.BatchNorm2d(2*self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 256*8*8
            nn.Conv2d(2*self.ndf, 4*self.ndf, kernel_size, 2, 2, bias=False),
            nn.BatchNorm2d(4*self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 512*4*4
            nn.Conv2d(4*self.ndf, 4*self.ndf, 4, 1, 0, bias=False)
            # 128*1*1
        )
        self.fullnn2 = FullNN_phi(4*h_dim, time_dim2, n_full_layers2)
        
        self.ode_func_net = create_net(ode_in_dim, ode_latent_dim)
        self.odefunc = ODEFunc(input_dim=ode_in_dim, latent_dim=ode_latent_dim, ode_func_net=self.ode_func_net)
        self.ode_solver = DiffeqSolver(input_dim=ode_in_dim, ode_func=self.odefunc, method='euler', latents=10, device=device)
        self.ode_solver_loss = DiffeqSolver_KL(device=device)

        self.rnn = SRU(input_size=ode_latent_dim, hidden_size=ode_latent_dim, num_layers=2)

        self.deconv = nn.Sequential(
            # 512*1*1
            nn.ConvTranspose2d(4*self.ndf, 4*self.ndf, 4, 4, 0, bias=False),
            nn.BatchNorm2d(4*self.ndf),
            nn.ReLU(inplace=True),
            # 512*4*4
            nn.ConvTranspose2d(4*self.ndf, 2*self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*self.ndf),
            nn.ReLU(inplace=True),
            # 256*8*8
            nn.ConvTranspose2d(2*self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.ReLU(inplace=True),
            # 128*16*16
            nn.ConvTranspose2d(self.ndf, in_dim, 4, 4, 0, bias=False),
            nn.Tanh()
            # 1*64*64
        )

    def forward(self, x, mode='train'):
        b, l ,c, w, h = x.shape
        x = x.reshape(b*l, c, w, h)
        zt_post = self.encode_xt(x)
        zt_post = zt_post.reshape(b, l, -1)
        input_x = torch.zeros((b, 1), requires_grad=True).to(self.device)
        integral1_list = []
        derivative1_list = []
        for i in range(l):
            integral1, derivative1 = self.fullnn1(
                input_x, zt_post[:, i, :], mode)
            integral1_list.append(integral1)
            derivative1_list.append(derivative1)
            input_x = integral1
        integral1_list = torch.stack(integral1_list, dim=1)
        derivative1_list = torch.stack(derivative1_list, dim=1)
        
        integral1_after_list = torch.cat((integral1_list[:, 1:, :], integral1_list[:, -1:, :]), dim=1)
      
        zt_prior = self.encode_prior(x)
        zt_prior = zt_prior.reshape(b, l, -1)
        input_x = torch.zeros((b, 1), requires_grad=True).to(self.device)
        integral2_list = []
        derivative2_list = []
        for i in range(l):
            integral2, derivative2 = self.fullnn2(
                input_x, zt_prior[:, i, :], mode)
            integral2_list.append(integral2)
            derivative2_list.append(derivative2)
            input_x = integral2
        integral2_list = torch.stack(integral2_list, dim=1)
        derivative2_list = torch.stack(derivative2_list, dim=1)

        # post q_zt
        ode_before = zt_post
        zt_inv_list = []
        for j in range(integral1_list.shape[1]):
            t_now = integral1_list[:, j, :]
            t_after = integral1_after_list[:, j, :]
            t_seq = torch.cat((t_now, t_after), dim=-1)
            zt_inv = self.ode_solver(zt_post[:, j, :], t_seq)
            zt_inv_list.append(zt_inv)
        qzt_inv = torch.stack(zt_inv_list, dim=1)
        input_rnn = qzt_inv.permute(1, 0, 2)
        h0 = torch.zeros((2, b, 4*self.ndf)).to(self.device)
        ht, _ = self.rnn(input_rnn, h0)
        ht = ht.permute(1, 0, 2)
        ode_after = qzt_inv
     
        qzt_inv = ht
        qzt_inv = qzt_inv.reshape(b*l, 4*self.ndf, 1, 1)
        qzt_deconv = self.deconv(qzt_inv)
        qzt_deconv = qzt_deconv.reshape(b, l, c, w, h) 

        kl2 = self.kl_pt_qt(integral1_list, derivative1_list,
                            integral2_list, derivative2_list)

        # reconstruction loss
        x = x.reshape(b, l, c, w, h)
        x_next = torch.cat((x[:, 1:, :, :, :], x[:, -1:, :, :, :]), dim=1)
        mse = torch.mean(F.mse_loss(qzt_deconv, x_next))
        loss = mse + opt.w_kl*kl2 

        return loss, mse, kl2, integral1_list.reshape(b, l)

    def kl_pt_qt(self, integral1, derivative1, integral2, derivative2):
        loss = torch.zeros((opt.train_batch_size, opt.video_length)).to(self.device)
        for i in range(opt.video_length):
            loss_i = self.ode_solver_loss(integral1[:, i, :], derivative1[:, i, :], integral2[:, i, :], derivative2[:, i, :])
            loss[:, i:i+1] = loss_i

        return torch.mean(loss).to(self.device)

