import os
import torch
from PIL import Image
class PreProcessData():
    def __init__(self, dir_path, transform=None):
        super().__init__()
        self.transform = transform
        self.ids=50
        self.data_path = dir_path
        self.file_names = [f for f in os.listdir(self.data_path)
                      if f.endswith('.jpg')]
        self.file_dict = dict()
        for f_name in self.file_names:
            fields = f_name.split('.')[0].split('_')
            identity = fields[0]
            head_pose = fields[2]
            side = fields[-1]
            key = '_'.join([identity, head_pose, side])
            if key not in self.file_dict.keys():
                self.file_dict[key] = []
                self.file_dict[key].append(f_name)
            else:
                self.file_dict[key].append(f_name)
        self.train_images = []
        self.train_angles_r = []
        self.train_labels = []
        self.train_images_t = []
        self.train_angles_g = []

        self.test_images = []
        self.test_angles_r = []
        self.test_labels = []
        self.preprocess()
    def preprocess(self):

        for key in self.file_dict.keys():

            if len(self.file_dict[key]) == 1:
                continue

            idx = int(key.split('_')[0])
            flip = 1
            if key.split('_')[-1] == 'R':
                flip = -1

            for f_r in self.file_dict[key]:

                file_path = os.path.join(self.data_path, f_r)

                h_angle_r = flip * float(
                    f_r.split('_')[-2].split('H')[0]) / 15.0
                    
                v_angle_r = float(
                    f_r.split('_')[-3].split('V')[0]) / 10.0
                    

                for f_g in self.file_dict[key]:

                    file_path_t = os.path.join(self.data_path, f_g)

                    h_angle_g = flip * float(
                        f_g.split('_')[-2].split('H')[0]) / 15.0
                        
                    v_angle_g = float(
                        f_g.split('_')[-3].split('V')[0]) / 10.0
                        

                    if idx <= self.ids:
                        self.train_images.append(file_path)
                        self.train_angles_r.append([h_angle_r, v_angle_r])
                        self.train_labels.append(idx - 1)
                        self.train_images_t.append(file_path_t)
                        self.train_angles_g.append([h_angle_g, v_angle_g])
                if idx > self.ids :
                    self.test_images.append(file_path)
                    self.test_angles_r.append([h_angle_r, v_angle_r])
                    self.test_labels.append(idx - 1)
                    
    def training_data(self):
            return self.train_images,self.train_angles_r,self.train_labels,self.train_images_t,self.train_angles_g
    def testing_data(self):
            return self.test_images,self.test_angles_r,self.test_labels
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, images,angles_r,labels,images_t,angles_g, transform=None):
        super().__init__()
        self.transform = transform
        self.images=images
        self.angles_r=angles_r
        self.labels=labels
        self.images_t=images_t
        self.angles_g=angles_g
    def __getitem__(self, index):
        return (
            self.transform(Image.open(self.images[index])),
                torch.tensor(self.angles_r[index]),
                self.labels[index],
            self.transform(Image.open(self.images_t[index])),
                torch.tensor(self.angles_g[index]))
        
    def __len__(self):
        return len(self.images)
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, images,angles_r,labels, transform=None):
        super().__init__()
        self.transform = transform
        self.images=images
        self.angles_r=angles_r
        self.labels=labels
    def __getitem__(self, index):
        return (
            self.transform(Image.open(self.images[index])),
                torch.tensor(self.angles_r[index]),
                self.labels[index])
        
    def __len__(self):
        return len(self.images)
