"""
a code to construct a 1d condition variational autoencoder (VAE) with pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DM(nn.Module):
    # a pre-defined forward dynamic system
    def __init__(self, ax, bx, ndof, dt, device=device):
        """
        ax -- (ndof, ) or init
        bx -- (ndof, ) or init
        ndof -- int
        dt -- float
        x0 -- (ndof, )
        min -- (ndof, ), the denormalization parameter
        max -- (ndof, ), the denormalization parameter
        """
        super(DM, self).__init__()

        self.device = device    
        self.ndof = ndof
        # constant parameter of a dynamic system
        self.dt = torch.tensor(dt).to(self.device)
        self.ax = torch.tensor(ax).to(self.device)
        self.bx = torch.tensor(bx).to(self.device)

        # define trainable scalar parameters
        self.scal = torch.zeros(ndof, device=self.device)

    def forward(self, force, goal=None, x0=None, dx=None, ddx=None):
        """
        force -- (batch, ndof, time_steps)
        goal -- (batch, ndof)
        dx -- (batch, ndof)
        ddx -- (batch, ndof)
        """
        if (x0 is None):
            self.x = torch.zeros(self.ndof, device=self.device)
        else: 
            self.x = torch.tensor(x0).to(self.device)

        batch, ndof, time  = force.shape

        # ---------- initial state ----------
        self.dx = torch.zeros(batch, ndof, device=self.device)
        self.ddx = torch.zeros(batch, ndof, device=self.device)
        
        traj = torch.zeros_like(force)
        if goal is None:
            goal = torch.zeros_like(self.x)
        else:
            goal = torch.tensor(goal).to(self.device) 
        
        # rollout the force in time_steps
        for i in range(time):            
            self.ddx = self.ax * (self.bx * (goal - self.x) - self.dx) + force[:, :, i]
            self.dx = self.dx + self.ddx * self.dt
            self.x = self.x + self.dx * self.dt
            traj[:, :, i] = self.x
        return traj


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation=True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                # q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self, shape, nhid=4, ncond=0):
        """
        the shape of input data -- (ndof, time_steps)
        """
        super(Encoder, self).__init__()
        self.n_dof, self.time_steps = shape
        # calculate the shape of the output of the encoder
        ww = ((self.time_steps - 5) // 2 + 1)
        ww = ((ww - 5) // 2 + 1) // 2
        ww = ((ww - 3) // 2 + 1)
        ww = ((ww - 3) // 2 + 1) // 2

        self.encode = nn.Sequential(nn.Conv1d(self.n_dof, 16, 5, stride=2, padding=0), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
                                    nn.Conv1d(16, 32, 5, stride=2, padding=0), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
                                    nn.MaxPool1d(2),
                                    nn.Conv1d(32, 64, 3, stride=2, padding=0), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                                    nn.Conv1d(64, 64, 3, stride=2, padding=0), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                                    nn.MaxPool1d(2),
                                    Flatten(), MLP([ww*64, self.n_dof*32]))

        self.calc_mean = MLP([32*self.n_dof+ncond, 16*self.n_dof, nhid*self.n_dof], last_activation=False)
        self.calc_logvar = MLP([32*self.n_dof+ncond, 8*self.n_dof, nhid*self.n_dof], last_activation=False)

    def forward(self, x, y=None):
        """
        :param x: (batch_size, n_dof, time_steps)
        :param y: (batch_size, ncond)  -- condition
        DMP: if use DMP decoder
        """
        x = self.encode(x)
        if (y is None):
            return self.calc_mean(x), self.calc_logvar(x)
        else:
            mean = self.calc_mean(torch.cat((x, y), dim=1))
            logvar = self.calc_logvar(torch.cat((x, y), dim=1))

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, shape, nhid=8, ncond=0):
        super(Decoder, self).__init__()
        self.n_dof, self.time_steps = shape
        self.shape = shape
        self.decode = nn.Sequential(MLP([nhid*self.n_dof+ncond, 64, 128, self.time_steps*self.n_dof],
                                        last_activation=False), nn.Sigmoid())

    def forward(self, z, y=None):
        c, w = self.shape
        if (y is None):
            return self.decode(z).view(-1, c, w)
        else:
            return self.decode(torch.cat((z, y), dim=1)).view(-1, c, w)


class CVAE(nn.Module):
    def __init__(self, shape, nclass, nhid=8, ncond=4):
        super(CVAE, self).__init__()
        self.n_dof, self.time_steps = shape
        self.dim = self.n_dof * nhid
        self.encoder = Encoder(shape, nhid, ncond)
        self.decoder = Decoder(shape, nhid, ncond)
        self.label_embedding = nn.Embedding(nclass, ncond)
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y=None):
        if (y is not None):
            y = self.label_embedding(y)
        mean, logvar = self.encoder(x, y)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z, y)
        return x_hat, mean, logvar

    def reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def loss(self, X, X_hat, mean, logvar):
        reconstruction_loss = self.mse_loss(X_hat, X)
        KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
        return reconstruction_loss + KL_divergence

    def generate(self, class_idx):
        if (type(class_idx) is int):
            class_idx = torch.tensor(class_idx)
        class_idx = class_idx.to(device)
        if (len(class_idx.shape) == 0):
            batch_size = None
            class_idx = class_idx.unsqueeze(0)
            z = torch.randn((1, self.dim)).to(device)
        else:
            batch_size = class_idx.shape[0]
            z = torch.randn((batch_size, self.dim)).to(device)

        y = self.label_embedding(class_idx)
        res = self.decoder(z, y)      
        
        return res

# fine-tune the decoder to generate the trajectory
class TrajGen(nn.Module):
    """
    Generate trajectory with torque and dm system 
    """
    def __init__(self, shape, nclass, min, max, nhid=8, ncond=4, device=None):
        """
        shape: (dof, time_steps)
        nclass: number of classes
        nhid: hidden dimension
        ncond: condition dimension, encode the trajectory type
        """
        super(TrajGen, self).__init__()
        self.n_dof, self.time_steps = shape
        self.dim = self.n_dof * nhid
        self.device = device

        # dm parameters
        self.dt = 0.01
        self.ax = 25.0
        self.bx = self.ax / 4.0

        # denormalization parameter
        self.min = min
        self.max = max
        if type(min) or type(max) is not torch.Tensor:
            self.min = torch.tensor(min).to(self.device)
            self.max = torch.tensor(max).to(self.device)
            self.min = self.min.unsqueeze(0).unsqueeze(-1)
            self.max = self.max.unsqueeze(0).unsqueeze(-1)

        self.label_embedding = nn.Embedding(nclass, ncond).to(device)

        self.decoder_o = Decoder(shape, nhid, ncond).to(device)

        # fix paramters in the decoder, label embedding
        for param in self.label_embedding.parameters():
            param.requires_grad = False

        for param in self.decoder_o.parameters():
            param.requires_grad = False

        # deep copy the decoder 
        self.decoder_n = copy.deepcopy(self.decoder_o)

        # release the final layer of the decoder to train
        d_mlp = self.decoder_n.decode[-2]
        d_mlp.mlp[-1].weight.requires_grad = True
        d_mlp.mlp[-1].bias.requires_grad = True   
        d_mlp.mlp[2].weight.requires_grad = True
        d_mlp.mlp[2].bias.requires_grad = True     

        self.dm = DM(self.ax, self.bx, self.n_dof, self.dt)  

    def forward(self, class_idx, x0, goal):
        traj = self.generator(class_idx, x0, goal, d_type="new")    
        return traj 

    def generator(self, class_idx, x0, goal, d_type="old"):
        # check data type is not tensor
        if (type(class_idx) is not torch.Tensor):
            class_idx = torch.tensor(class_idx).to(self.device)
        
        if (len(class_idx.shape) == 0):
            batch_size = 1
            class_idx = class_idx.unsqueeze(0)
            
        else:
            batch_size = class_idx.shape[0]

        z = torch.randn((batch_size, self.dim)).to(device)

        y = self.label_embedding(class_idx) 

        if (d_type == "old"):       
            res = self.decoder_o(z, y)
        elif (d_type == "new"):
            res = self.decoder_n(z, y)

        # put the generated force to DM system
        torque = res * (self.max - self.min) + self.min
        traj = self.dm(force =torque, goal=goal, x0=x0)

        return traj
    
    def loss(self, point, class_idx, x0, goal):
        """
        point: (batch_size, n_dof) -- the via point must get through
        class_idx: (batch_size, ) -- the class of trajectory
        x0: (n_dof) -- the initial state of the trajectory
        goal: (n_dof) -- the goal of the trajectory
        """
        o_traj = self.generator(class_idx, x0, goal, d_type="old")
        n_traj = self.generator(class_idx, x0, goal, d_type="new")
        
        # via point to the trajectory
        if point is None:
            p_loss = 0
        else:
            p_loss = torch.min(torch.norm(n_traj - point, p=2, dim=1))

        # trajectory shape loss
        s_loss = torch.mean(torch.norm(o_traj - n_traj, p=2, dim=1))
        # end position loss
        e_loss = torch.norm(n_traj[:, :, -1] - goal, p=2, dim=1)
        return p_loss + s_loss * 0.2 + 0.2 * e_loss    

# generate trajectory from torque
class DmpCVAE(CVAE):
    def __init__(self, dmp, shape, nclass, nhid=8, ncond=4):
        super(DmpCVAE, self).__init__(shape, nclass, nhid, ncond)
        self.dmp = dmp

    def get_trajectory(self, torque, y0, goal):
        """
        :param torque: (batch_size, n_dof, time_steps)
        :param y0: (batch_size, n_dof)
        :param goal: (batch_size, n_dof)
        """        
        if (len(torque.shape) == 2):
            torque = torque[np.newaxis, :, :]
            y0 = y0[np.newaxis, :]
            goal = goal[np.newaxis, :]

        batches, dof, steps = torque.shape
        trajectory = np.zeros(torque.shape)
        for i in range(batches):
            for j in range(dof):
                trajectory[i, j, :], _, _ = self.dmp.torque_rollout(torque[i, j, :], y0[i, j], goal[i, j])
        return trajectory