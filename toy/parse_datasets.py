
import torch
from torch.utils.data import Dataset

import numpy as np
from config import opt

def parse_datasets(opt, mode, dataset, device):
    if opt.dataset_name == 'sin':
        data_obj = SinDataset(opt, mode, dataset, device)
        return data_obj

class SinDataset(Dataset):
    def __init__(self, opt, mode, dataset, device):
        self.opt = opt
        self.mode = mode
        self.dataset = dataset
        self.device = device
        if opt.isLoad == True:
            if mode == 'train':
                self.data = torch.load('./data/load_'+str(mode)+'_'+self.dataset+'.pt')
            elif mode == 'test':
                self.data = torch.load('./data/load_'+str(mode)+'_'+self.dataset+'.pt')
            print("Load "+self.dataset+'_'+mode+"set successfully!")
    
        else:
            if self.dataset == 'pois':
                self.poisson_process = poisson_process(lambd=10.)
            elif self.dataset == 'hks':
                self.hawkes_process = MHP(mu=[10.])
            self.data = self.sample_sin_value()
            self.save_data(self.data, './data/load_'+str(mode)+'_'+self.dataset+'.pt')
            print("Generate and save "+self.dataset+'_'+mode+"set successfully!")

    def save_data(self, data, path):
        torch.save(data, path)

    def sample_sin_value(self, z0=1., noise_weight=0.1):
        res_list = []
        
        for i in range(opt.n_samples[self.mode]):
            if self.dataset == 'pois':
                time_seq = self.poisson_process.generate_seq(self.opt.n_total_tp)
            elif self.dataset == 'hks':
                time_seq = self.hawkes_process.generate_seq(self.opt.n_total_tp)
                time_seq = [t[0] for t in time_seq]

            res = []
            phi = 0.
            amp = 1.
            freq = 1.
            for t in time_seq:
                phi = 2*np.pi*freq*t
                x = amp*np.sin(phi) 
                res.append([x, t])
            res = np.array(res)
            res_list.append(res)

        res_list = np.array(res_list)
        res_list = torch.tensor(res_list, dtype=torch.float32).to(self.device)
        res_list = self.add_noise(res_list, noise_weight)

        return res_list

    def add_noise(self, res_list, noise_weight):
        n_samples = res_list.size(0)
        # Add noise to all the points except the first point
        n_tp = self.opt.n_total_tp - 1
        noise = np.random.sample((n_samples, n_tp))
        noise = torch.tensor(noise, dtype=torch.float32).to(self.device)

        res_list_w_noise = res_list.clone()
        res_list_w_noise[:, 1:, 0] += noise_weight * noise
        # print(res_list_w_noise[:, :, 0])
        return res_list_w_noise

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        return self.data[idx], 0



class poisson_process:
    def __init__(self, lambd):
        self.lambd = lambd

    def generate_seq(self, size):
        self.data = []

        scale = 1./self.lambd
        s = 0
        # self.data.append(s)
        for i in range(size):
            w = 0
            w = np.random.exponential(scale=scale)
            s += w
            self.data.append(s)

        return self.data


class MHP:
    def __init__(self, alpha=[[0.5]], mu=[0.1], omega=1.0):
        '''params should be of form:
        alpha: numpy.array((u,u)), mu: numpy.array((,u)), omega: float'''

        self.data = []
        self.alpha, self.mu, self.omega = np.array(alpha), np.array(mu), omega
        self.dim = self.mu.shape[0]
        self.check_stability()

    def check_stability(self):
        ''' check stability of process (max alpha eigenvalue < 1)'''
        w, v = np.linalg.eig(self.alpha)
        me = np.amax(np.abs(w))
        # print('Max eigenvalue: %1.5f' % me)
        if me >= 1.:
            print('(WARNING) Unstable.')

    def generate_seq(self, size):
        '''Generate a sequence based on mu, alpha, omega values. 
        Uses Ogata's thinning method, with some speedups, noted below'''

        self.data = []  # clear history

        Istar = np.sum(self.mu)
        s = 0.
        s = np.random.exponential(scale=1./Istar)

        # attribute (weighted random sample, since sum(mu)==Istar)
        n0 = np.random.choice(np.arange(self.dim),
                              1,
                              p=(self.mu / Istar))
        self.data.append([s, n0])

        # value of \lambda(t_k) where k is most recent event
        # starts with just the base rate
        lastrates = self.mu.copy()

        decIstar = False
        while len(self.data) < size:
            tj, uj = self.data[-1][0], int(self.data[-1][1])

            if decIstar:
                # if last event was rejected, decrease Istar
                Istar = np.sum(rates)
                decIstar = False
            else:
                # otherwise, we just had an event, so recalc Istar (inclusive of last event)
                Istar = np.sum(lastrates) + \
                    self.omega * np.sum(self.alpha[:, uj])

            # generate new event
            w = 0.
            w = np.random.exponential(scale=1./Istar)
            s += w

            # calc rates at time s (use trick to take advantage of rates at last event)
            rates = self.mu + np.exp(-self.omega * (s - tj)) * \
                (self.alpha[:, uj].flatten() *
                 self.omega + lastrates - self.mu)

            # attribution/rejection test
            # handle attribution and thinning in one step as weighted random sample
            diff = Istar - np.sum(rates)
            try:
                n0 = np.random.choice(np.arange(self.dim+1), 1,
                                      p=(np.append(rates, diff) / Istar))
            except ValueError:
                # by construction this should not happen
                print('Probabilities do not sum to one.')
                self.data = np.array(self.data)
                return self.data

            if n0 < self.dim:
                self.data.append([s, n0])
                # update lastrates
                lastrates = rates.copy()
            else:
                decIstar = True

        self.data = np.array(self.data)       
        return self.data

