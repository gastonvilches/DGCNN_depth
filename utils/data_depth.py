import os
import numpy as np
from torch.utils.data import Dataset
import random
from scipy.sparse import load_npz

class DepthDataset(Dataset):
    def __init__(self, partition='train', outputs=['depth','feature_map','edge_label','edge_index'], 
                 models=[], shuffle_pixels=False, pc_mean=None, repeat=1, preload=False):
        
        super().__init__()
        
        self.pc_mean = pc_mean
        self.outputs = outputs
        self.dir = 'C:/Users/Gastón/Desktop/Gaston/CONICET/repos/dgcnn_emma/data/data_2k_edgelabels_fmap_5models_d15'
        # self.dir = '/media/gaston/Windows-SSD/Users/Gastón/Desktop/Gaston/CONICET/repos/dgcnn_emma/data'
        
        if partition == 'train':
            self.dir += '_train'
        elif partition[:5] == 'valid':
            self.dir += '_valid'
        
        self.file_list = [file for file in os.listdir(self.dir) if len(file.split('.'))==2]
        self.model_list = [file.split('_')[0] for file in self.file_list]
        
        # Retain only files present in input models list
        assert all([model in self.model_list for model in models]), 'At least one input model is not in the specified partition'
        if len(models) > 0:
            file_list_temp = []
            for file in self.file_list:
                if file.split('_')[0] in models:
                    file_list_temp.append(file)
            self.file_list = file_list_temp
            print(len(file_list_temp))
        
        first_file = np.load(os.path.join(self.dir, self.file_list[0]))
        self.im_shape = first_file['im_shape'][:,np.newaxis]
        self.repeat = repeat
        
        self.shuffle_pixels = shuffle_pixels
        
        self.preload = preload
        if self.preload:
            self.data = []
            for i in range(len(self)):
                self.data.append(self._load_file(i))
        
    def _load_file(self, item):
        # item = 0
        item = item // self.repeat
        file = os.path.join(self.dir, self.file_list[item])
        data = np.load(file)
        return data
        
    def __getitem__(self, item):
        data = self.data[item] if self.preload else self._load_file(item)
        pixels = data['pixels']
        if 'depth' in self.outputs:
            depth = data['depth']
        if 'feature_map' in self.outputs:
            feature_map = data['feature_map']
        if 'edge_label' in self.outputs:
            edge_label = data['edge_label_NxK'].astype(np.int64)
        if 'edge_index' in self.outputs:
            edge_index = data['edge_index'].astype(np.int64)
        
        # Unison shuffle
        if self.shuffle_pixels:
            shuffler = random.sample(range(pixels.shape[1]), pixels.shape[1])
            pixels = pixels[:,shuffler]
        output = [pixels]
        if 'depth' in self.outputs:
            if self.shuffle_pixels:
                depth = depth[shuffler]
            if self.pc_mean != None:
                depth += self.pc_mean - depth.mean()
            output.append(depth)
        if 'feature_map' in self.outputs:
            if self.shuffle_pixels:
                feature_map = feature_map[:,shuffler,:]
            output.append(feature_map)
        if 'edge_label' in self.outputs:
            if self.shuffle_pixels:
                
                
                
                print('EL SHUFFLER NO ANDA EN EDGE LABEL')
                edge_label = edge_label[shuffler,:][:,shuffler]
                
                
                
            output.append(edge_label)
        if 'edge_index' in self.outputs:
            if self.shuffle_pixels:
                
                
                
                print('NO SE IMPLEMENTÖ SHUFFLER ACA')
                pass
            
            
            
            output.append(edge_index)
        return output
        
    def __len__(self):
        # return 10
        return len(self.file_list)*self.repeat
    
    def denormalize(self, pixels):
        pixels = (pixels + 1) * self.im_shape / 2
        return pixels.astype(np.int32)

class DepthDatasetFm(Dataset):
    def __init__(self, partition='train', models=[], num_points=None, pc_mean=None, repeat=1):
        
        self.pc_mean = pc_mean
        self.dir = 'C:/Users/Gastón/Desktop/Gaston/CONICET/repos/dgcnn_emma/data/data_2k_fmap_5models_d15'
        # self.dir = '/media/gaston/Windows-SSD/Users/Gastón/Desktop/Gaston/CONICET/repos/dgcnn_emma/data'
        
        if partition == 'train':
            self.dir += '_train'
        elif partition[:5] == 'valid':
            self.dir += '_valid'
        
        self.file_list = [file for file in os.listdir(self.dir) if len(file.split('.'))==2]
        self.model_list = [file.split('_')[0] for file in self.file_list]
        
        # Retain only files present in input models list
        assert all([model in self.model_list for model in models]), 'At least one input model is not in the specified partition'
        if len(models) > 0:
            file_list_temp = []
            for file in self.file_list:
                if file.split('_')[0] in models:
                    file_list_temp.append(file)
            self.file_list = file_list_temp
            print(len(file_list_temp))
        
        first_file = np.load(os.path.join(self.dir, self.file_list[0]))
        self.im_shape = first_file['im_shape'][:,np.newaxis]
        self.num_points = num_points
        self.repeat = repeat
        
    def __getitem__(self, item):
        # item = 0
        item = item // self.repeat
        file = os.path.join(self.dir, self.file_list[item])
        data = np.load(file)
        pixels = data['pixels']
        depth = data['depth']
        feature_map = data['feature_map']
        
        # Unison shuffle
        num_points = depth.shape[0]
        shuffler = random.sample(range(depth.shape[0]), num_points)
        p = pixels[:,shuffler]
        d = depth[shuffler]
        f = feature_map[:,shuffler,:]
        
        # Center mean
        if self.pc_mean != None:
            d += self.pc_mean - d.mean()
        
        return p, d, f

    def __len__(self):
        # return 10
        return len(self.file_list)*self.repeat
    
    def denormalize(self, pixels):
        pixels = (pixels + 1) * self.im_shape / 2
        return pixels.astype(np.int32)

class DepthDatasetOld(Dataset):
    def __init__(self, partition='train', models=[], num_points=None, normalize=True, pc_mean=None, repeat=1):
        
        self.pc_mean = pc_mean
        self.dir = 'C:/Users/Gastón/Desktop/Gaston/CONICET/repos/dgcnn_emma/data/data_2k'
        # self.dir = '/media/gaston/Windows-SSD/Users/Gastón/Desktop/Gaston/CONICET/repos/dgcnn_emma/data'
        
        if partition == 'train':
            self.dir += '_train'
        elif partition[:5] == 'valid':
            self.dir += '_valid'
        
        self.file_list = [file for file in os.listdir(self.dir) if len(file.split('.'))==2]
        self.model_list = [file.split('_')[0] for file in self.file_list]
        
        # Retain only files present in input models list
        assert all([model in self.model_list for model in models]), 'At least one input model is not in the specified partition'
        if len(models) > 0:
            file_list_temp = []
            for file in self.file_list:
                if file.split('_')[0] in models:
                    file_list_temp.append(file)
            self.file_list = file_list_temp
            print(len(file_list_temp))
        
        first_file = np.load(os.path.join(self.dir, self.file_list[0]))
        self.im_shape = first_file['im_shape'][:,np.newaxis]
        self.num_points = num_points
        self.normalize = normalize
        self.repeat = repeat
        
    def __getitem__(self, item):
        # item = 0
        item = item // self.repeat
        file = os.path.join(self.dir, self.file_list[item])
        data = np.load(file)
        pixels = data['pixels']
        depth = data['depth'].squeeze()
        
        # Scale and center
        if self.normalize:
            pixels = 2 * pixels.astype(np.double) / self.im_shape - 1
        
        # Unison shuffle
        num_points = self.num_points if self.num_points != None else depth.shape[0]
        shuffler = random.sample(range(depth.shape[0]), num_points)
        p = pixels[:,shuffler]
        d = depth[shuffler]
        
        # Center mean
        if self.pc_mean != None:
            d += self.pc_mean - d.mean()
        
        return p.astype(np.float32), d.astype(np.float32)/255

    def __len__(self):
        # return 10
        return len(self.file_list)*self.repeat
    
    def denormalize(self, pixels):
        pixels = (pixels + 1) * self.im_shape / 2
        return pixels.astype(np.int32)






# Deprecated:

class DepthKNNDataset(Dataset): 
    def __init__(self, num_points=None, normalize=True, partition='train', repeat=1):
        
        self.dir = 'C:/Users/Gastón/Desktop/Gaston/CONICET/repos/dgcnn_emma/data_2k_k60_std4'
        # self.dir = '/media/gaston/Windows-SSD/Users/Gastón/Desktop/Gaston/CONICET/repos/dgcnn_emma/data'
        
        if partition == 'train':
            self.dir += '_train'
        elif partition[:5] == 'valid':
            self.dir += '_valid'
        
        self.file_list = [file for file in os.listdir(self.dir) if len(file.split('.'))==2]
        
        first_file = np.load(os.path.join(self.dir, self.file_list[0]))
        self.k = int(first_file['k'])
        self.im_shape = first_file['im_shape'][:,np.newaxis]
        self.num_points = num_points
        self.normalize = normalize
        self.repeat = repeat
        
    def __getitem__(self, item):
        # item = 0
        item = item // self.repeat
        file = os.path.join(self.dir, self.file_list[item])
        data = np.load(file)
        pixels = data['pixels']
        depth = data['depth'].squeeze()
        knn = load_npz(file.split('.')[0] + '.knn.npz').toarray()
        
        # Scale and center
        if self.normalize:
            pixels = 2 * pixels.astype(np.double) / self.im_shape - 1
        
        # Unison shuffle
        num_points = self.num_points if self.num_points != None else depth.shape[0]
        shuffler = random.sample(range(depth.shape[0]), num_points)
        p = pixels[:,shuffler]
        d = depth[shuffler]
        knn = knn[shuffler,:][:,shuffler]
        
        return p.astype(np.float32), d.astype(np.float32)/255, knn.astype(np.float32)

    def __len__(self):
        # return 10
        return len(self.file_list)*self.repeat
    
    def denormalize(self, pixels):
        pixels = (pixels + 1) * self.im_shape / 2
        return pixels.astype(np.int32)