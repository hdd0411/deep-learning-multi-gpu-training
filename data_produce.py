import numpy as np
import h5py
from config import *

class data_produce(object):
    def __init__(self,is_train):
        self.is_train=is_train
        self.path=get_config(self.is_train).data_path
        self.batch_size=get_config(self.is_train).batch_size
        self.filename=get_config(self.is_train).filename


    def load_data(self):
        dataset = h5py.File(self.path+'/'+self.filename, 'r')
        x = np.array(dataset['data']).transpose((0,2,3,1))
        y = np.array(dataset['label']).transpose((0,2,3,1))
        return x, y

    def minibatches_produce(self,X,Y, seed):
        m = X.shape[0]
        batch_total = int(m / self.batch_size)
        np.random.seed(seed)
        index = list(np.random.permutation(m))
        X_shuffle = X[index, :, :, :]
        Y_shuffle = Y[index, :, :, :]
        minibatches = []
        for start_id in range(batch_total):
            minibatch_x = X_shuffle[start_id * self.batch_size:start_id * self.batch_size + self.batch_size, :, :, :]
            minibatch_y = Y_shuffle[start_id * self.batch_size:start_id * self.batch_size + self.batch_size, :, :, :]
            minibatch = (minibatch_x, minibatch_y)
            minibatches.append(minibatch)
        if m % self.batch_size != 0:
            minibatch_x = X_shuffle[self.batch_size * batch_total:m, :, :, :]
            minibatch_y = Y_shuffle[self.batch_size * batch_total:m, :, :, :]
            minibatch = (minibatch_x, minibatch_y)
            minibatches.append(minibatch)
        return minibatches


