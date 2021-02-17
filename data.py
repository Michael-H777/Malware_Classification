from configs import *
from packages import *


class DataScaler:
    
    def __init__(self, *, domain_min=0, domain_max=0, target_min=-1, target_max=1):
        self.domain_min = domain_min
        self.domain_range = domain_max - domain_min
        
        self.target_min = target_min
        self.target_range = target_max - target_min
        
    def get_domain_stats(self, folder_path):
        domain_min, domain_max = np.inf, -np.inf
        for filename in os.listdir(folder_path):
            data = tifffile.imread(f'{folder_path}/{filename}')
            domain_min = np.min(domain_min, data.min())
            domain_max = np.max(domain_max, data.max())
        # update stats
        self.domain_min = domain_min
        self.domain_range = domain_max - domain_min
        return None 

    def to_target(self, input_data):
        position = (input_data - self.domain_min) / (self.domain_range)
        result = (self.target_range * position) + self.target_min
        return np.float32(result)
    
    def to_domain(self, input_data): 
        position = (input_data - self.target_min) / self.target_range
        result = position * self.domain_range + self.domain_min
        return np.float32(result)


def create_h5_dataset(patches, name_header, h5file, augment_times=0):
    pass


def make_model_data(*, augment_times=0, train_size=1500, val_size=50): 
    pass


def augment_data(image, mode):
    # all rotation is counter clock wise 
    out = image
    if mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out)


