from configs import *
from packages import * 

import data 


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def cos_annealing_lr(initial_lr, cur_epoch, epoch_per_cycle):
    return initial_lr * (np.cos(np.pi * cur_epoch / epoch_per_cycle) + 1) / 2


class make_loss_function:

    def __init__(self):
        pass

    def compute_loss(self, prediction, target_output): 
        pass


def main():
    pass


if __name__ == '__main__':
    
    if pre_process:
        data.make_model_data(augment_times=augment_times, train_size=train_size, val_size=val_size)

    main()