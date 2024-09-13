"""
a class to load trajectory and interpolate the data for DMP
"""
# add parent path to system path
import os.path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from scipy.interpolate import interp1d
import re
import torch
from torch.utils.data import Dataset
from models.dmp import CanonicalSystem, SingleDMP
from utils.utils import quaternion_to_yaw, remove_close_position, smooth_position   # pre-processing the data

class DataLoader(Dataset):
    """
    load the data from the file
     1. data augmentation
     2. normalization the task
    """
    def __init__(self, run_time, dt, dof, dmp):
        self.run_time = run_time
        self.dt = dt
        self.time_steps = np.arange(0, self.run_time, self.dt)
        self.dof = dof
        self.paths = []
        self.labels = []
        self.cs = CanonicalSystem(dt=0.01, ax=1)
        self.dmp = dmp

    def interpolation_data(self, traj):
        """
        load a data in [steps, dof] format
        """
        traj_time = np.linspace(0, self.run_time, traj.shape[0])
        assert traj.shape[1] == self.dof, "dof of data is not equal to the dof of DMP system"

        new_traj = np.zeros((len(self.time_steps), self.dof))
        for _d in range(self.dof):
            path_gen = interp1d(traj_time, traj[:, _d])
            new_traj[:, _d] = path_gen(self.time_steps)
        return new_traj

    def load_raw_data(self, input_name):
        f = open(input_name, 'r')
        points = []
        for row in f:
            points.append(row.strip('\n').split(','))
        f.close()

        points = np.array(points, dtype='float')
        points = points[:, :2]

        # need to rotate the points
        theta = np.pi / 2.
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        for ii in range(points.shape[0]):
            points[ii] = np.dot(R, points[ii])

        # need to mirror the x values
        for ii in range(points.shape[0]):
            points[ii, 0] *= -1

        # center numbers to -1 1
        points[:, 0] -= np.min(points[:, 0])
        points[:, 1] -= np.min(points[:, 1])

        # normalize
        points[:, 0] /= max(points[:, 0])
        points[:, 1] /= max(points[:, 1])

        if '1' in input_name:
            points[:, 0] /= 15.

        if '9' in input_name:
            points[:, 0] /= 2.
        return points

    def load_data_all(self, dict_name):
        """
        load data from a file
        """
        for filename in os.listdir(dict_name):

            if filename.endswith(".txt"):
                input_name = os.path.join(dict_name, filename)
                traj = self.load_raw_data(input_name)
                # create labels of the trajectory -- number, start, end
                number = int(re.findall(r'\d+', input_name)[0])
                labels = np.array([number])
                labels = np.append(labels, traj[0])
                labels = np.append(labels, traj[-1])
                # append trajectory and labels pairs
                self.paths.append([self.interpolation_data(traj), labels])

    def data_augment(self, number, weight_random=0.5):
        """
        data augmentation using classical DMP
        number: number of augmented data for each trajectory
        weight_random:  random noise for the weights of DMP
        """
        all_paths = []
        all_labels = []
        init_size = len(self.paths)
        aug_paths = self.paths.copy()

        for i in range(init_size):
            traj, label = aug_paths[i]
            new_paths = np.zeros((len(self.time_steps), number, self.dof))
            for j in range(self.dof):

                self.dmp.imitate_path(traj[:, j])
                w0 = self.dmp.w.detach()
                for k in range(number):
                    self.dmp.w = w0 + w0*(2*torch.rand(size=self.dmp.w.shape)-1)*weight_random
                    new_paths[:, k, j], _, _ = self.dmp.rollout(y0=traj[0, j], goal=traj[-1, j])

            for x in range(number):
                # update the label start and end points
                labels = np.array([label[0]])
                labels = np.append(labels, new_paths[0, x, :])
                labels = np.append(labels, new_paths[-1, x, :])
                all_paths.append(new_paths[:, x, :])
                all_labels.append(labels)

        self.paths = all_paths
        self.labels = all_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        traj, label = self.paths[idx], self.labels[idx]
        x = torch.tensor(traj.T, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.int)
        return x, y

    # save the data of paths
    def save_data(self, file_name):
        np.savez(file_name, paths=self.paths, lables=self.labels)

    # load the data of paths
    def load_data(self, file_name):
        loaded_data = np.load(file_name, allow_pickle=True)
        self.paths= loaded_data["paths"]
        self.labels = loaded_data["lables"]         

class TorqueLoader(DataLoader):
    def __init__(self, dmp, run_time=1.0, dt=0.01, dof=2):
        super(TorqueLoader, self).__init__(run_time, dt, dof, dmp)
        self.torque = None
        self.torque_labels = None
        self.torque_norm = None   
        
    def load_data(self, file_name, device):
        loaded_data = np.load(file_name, allow_pickle=True)
        self.torque = loaded_data["torque"]
        self.torque_labels = loaded_data["lables"]

        # convert to tensor
        self.torque = torch.tensor(self.torque, dtype=torch.float, device=device)
        self.torque_labels = torch.tensor(self.torque_labels, dtype=torch.int, device=device)

    def normalize_data(self, device):
        """
        data -- (time_step, batch, dof)
        """
        # check data type
        if isinstance(self.torque, np.ndarray):
            self.torque = torch.tensor(self.torque, dtype=torch.float, device=device)

        self.max = self.torque.max(dim=0).values.max(dim=0).values
        self.min = self.torque.min(dim=0).values.min(dim=0).values
        self.torque_norm = (self.torque - self.min) / (self.max - self.min)       

    def denormalize_data(self, data):
        """
        data -- (batch, dof, time_step)        
        """       
        if len(self.max.shape) == 1: 
            self.max = self.max.unsqueeze(0).unsqueeze(-1)
            self.min = self.min.unsqueeze(0).unsqueeze(-1) 
            
        return (data * (self.max - self.min) + self.min).detach()

    def gen_torque_dataset(self):
        # generate the dataset of torque
        self.torque = np.zeros((len(self.time_steps), len(self.paths), self.dof))
        self.torque_labels = []

        for i in range(len(self.paths)):
            traj, label = self.paths[i], self.labels[i]
            # generate the torque
            for j in range(self.dof):
                # generate the torque
                self.dmp.imitate_path(traj[:, j])
                self.torque[:, i, j] = self.dmp.gen_force()
            self.torque_labels.append(label)

    def __len__(self):
        # unnormlized data
        if self.torque_norm is None:
            return self.torque.shape[1]
        else:
            return self.torque_norm.shape[1]

    def __getitem__(self, idx):
        if self.torque_norm is None:
            torque, label = self.torque[:, idx, :], self.torque_labels[idx]  
        else:
            torque, label = self.torque_norm[:, idx, :], self.torque_labels[idx]     
        return torque.T, label.T

    # save the data of paths
    def save_data(self, file_name):
        np.savez(file_name, torque=self.torque, lables=self.torque_labels)


class ManipulationLoader(DataLoader):
    
    # load the data from .npy file
    def load_raw_data(self, file_name):  
        data = np.load(file_name, allow_pickle=True)
        return data
    
    def load_data_all(self, dict_name):
        # load all the .npy file in the folder
        for file in os.listdir(dict_name):
            if file.endswith(".npy"):
                file_name = os.path.join(dict_name, file)
                traj = self.load_raw_data(file_name)
                # task id is the first number in the file name
                number = int(file.split("_")[0])
                labels = np.array([number])
                labels = np.append(labels, traj[0])
                labels = np.append(labels, traj[-1])
                self.paths.append([self.interpolation_data(traj), labels])    

class ManipulationTorqueLoader(ManipulationLoader): 
    def __init__(self, dmp, run_time=1.0, dt=0.01, dof=2):
        super(ManipulationTorqueLoader, self).__init__(run_time, dt, dof, dmp)
        self.torque = None
        self.torque_labels = None
        self.torque_norm = None

    def gen_torque_dataset(self):
        # generate the dataset of torque
        self.torque = np.zeros((len(self.time_steps), len(self.paths), self.dof))
        self.torque_labels = []

        for i in range(len(self.paths)):
            traj, label = self.paths[i], self.labels[i]
            # generate the torque
            for j in range(self.dof):
                # generate the torque
                self.dmp.imitate_path(traj[:, j])
                self.torque[:, i, j] = self.dmp.gen_force()
            self.torque_labels.append(label)   
        
    def load_data(self, file_name, device="cpu"):
        loaded_data = np.load(file_name, allow_pickle=True)
        self.torque = loaded_data["torque"]
        self.torque_labels = loaded_data["lables"]

        # convert to tensor
        self.torque = torch.tensor(self.torque, dtype=torch.float, device=device)
        self.torque_labels = torch.tensor(self.torque_labels, dtype=torch.int, device=device)

    def normalize_data(self, device='cpu'):
        """
        data -- (time_step, batch, dof)
        """
        # check data type
        if isinstance(self.torque, np.ndarray):
            self.torque = torch.tensor(self.torque, dtype=torch.float, device=device)

        self.max = self.torque.max(dim=0).values.max(dim=0).values
        self.min = self.torque.min(dim=0).values.min(dim=0).values
        self.torque_norm = (self.torque - self.min) / (self.max - self.min)       

    def denormalize_data(self, data):
        """
        data -- (batch, dof, time_step)        
        """       
        if len(self.max.shape) == 1: 
            self.max = self.max.unsqueeze(0).unsqueeze(-1)
            self.min = self.min.unsqueeze(0).unsqueeze(-1) 
            
        return (data * (self.max - self.min) + self.min).detach()
    
    def __len__(self):
        # unnormlized data
        if self.torque_norm is None:
            return self.torque.shape[1]
        else:
            return self.torque_norm.shape[1]

    def __getitem__(self, idx):
        if self.torque_norm is None:
            torque, label = self.torque[:, idx, :], self.torque_labels[idx]  
        else:
            torque, label = self.torque_norm[:, idx, :], self.torque_labels[idx]     
        return torque.T, label.T

    # save the data of paths
    def save_data(self, file_name):
        np.savez(file_name, torque=self.torque, lables=self.torque_labels)


if __name__ == "__main__":
    # test data loader class
    cs = CanonicalSystem(dt=0.01, ax=1)
    dmp = SingleDMP(n_bfs=50, cs=cs, run_time=1.0, dt=0.01)

    # ------------------------------------ test the data loader ------------------------------------
    traj_loader = DataLoader(run_time=1.0, dmp=dmp, dt=0.01, dof=2)
    traj_loader.load_data_all("./data/")
    
    inter, _ = traj_loader.paths[0]
    
    plt.figure(2, figsize=(6, 6))
    plt.plot(inter[:, 0], inter[:, 1], "b")
    plt.title("DMP system - draw number interpolated")
    plt.show()
    
    # test data augmentation
    traj_loader.data_augment(number=1000, weight_random=0.3)
    print(len(traj_loader))
    traj_loader.save_data("./data/test_paths")
    
    # plot the augmented data
    plt.figure(2, figsize=(6, 6))
    # load all the data in the loader
    for i in range(len(traj_loader)):
        inter, label = traj_loader.paths[i], traj_loader.labels[i]
        if label[0] == 0:
            plt.plot(inter[:, 0], inter[:, 1])
    
    plt.axis("equal")
    plt.show() 

    # -------------------------------- test the torque loader -------------------------
    torque_loader = TorqueLoader(dmp=dmp, run_time=1.0, dt=0.01, dof=2)
    torque_loader.load_data_all("./data/")
    torque_loader.data_augment(number=5000, weight_random=0.3)

    # plot the augmented data
    check_number = 2
    plt.figure(2, figsize=(6, 6))
    # load all the data in the loader
    for i in range(len(torque_loader)):
        inter, label = torque_loader.paths[i],torque_loader.labels[i]
        if label[0] == check_number:
            plt.plot(inter[:, 0], inter[:, 1])

    plt.axis("equal")
    plt.show()

    torque_loader.gen_torque_dataset()
    torque_loader.save_data("./data/train_torque")

    torque_loader.torque = torque_loader.normalize_data(torque_loader.torque)

    # plot the augmented torque data
    plt.figure(3, figsize=(6, 6))
    for i in range(len(torque_loader)):
        inter, label = torque_loader.torque[:, i, :], torque_loader.torque_labels[i]
        if label[0] == 0:
            plt.plot(inter[:, 0])
            plt.plot(inter[:, 1])
    plt.show()

    # plot the denormalized torque data
    plt.figure(4, figsize=(6, 6))
    for i in range(len(torque_loader)):
        inter, label = torque_loader.denormalize_data(torque_loader.torque[:, i, :]), \
                                     torque_loader.torque_labels[i]
        if label[0] == 0:
            plt.plot(inter[:, 0])
            plt.plot(inter[:, 1])
    plt.show()