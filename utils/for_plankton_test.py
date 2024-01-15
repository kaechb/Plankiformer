###########
# IMPORTS #
###########

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import pickle

# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)


class CreateDataForPlankton:
    def __init__(self):
        self.classes = None
        self.classes_int = None
        self.Filenames = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.train_dataloader = None
        self.checkpoint_path = None
        self.y_val = None
        self.y_test = None
        self.y_train = None
        self.testFilenames = None
        self.trainFilenames = None
        self.valFilenames = None
        self.X_val = None
        self.X_test = None
        self.X_train = None
        self.class_weights_tensor = None
        self.params = None
        return

    def make_train_test_for_model(self, train_main, test_main, prep_data):
        Data = prep_data.Data
        # self.class_weights_tensor = prep_data.tt.class_weights_tensor
        # self.Filenames = prep_data.Filenames
        self.classes = np.load(test_main.params.main_param_path + '/classes.npy')
        self.Filenames = prep_data.Filenames
        self.class_weights_tensor = torch.load(test_main.params.main_param_path + '/class_weights_tensor.pt')

        if train_main.params.ttkind == 'mixed':
            self.trainFilenames = Data[0]
            trainXimage = Data[1]
            trainXfeat = Data[9]
            trX = [trainXimage, trainXfeat]
        elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'yes':
            self.trainFilenames = Data[0]
            trainXimage = Data[1]
            trainXfeat = Data[9]
            trX = [trainXimage, trainXfeat]
        elif train_main.params.ttkind == 'feat':
            self.trainFilenames = Data[0]
            trainXfeat = Data[9]
            trX = [trainXfeat]
        else:
            self.trainFilenames = Data[0]
            trX = Data[1]

        data_train = trX.astype(np.float64)
        data_train = 255 * data_train
        self.X_train = data_train.astype(np.uint8)

        return

    def make_train_test_for_model_with_y(self, train_main, test_main, prep_data):
        Data = prep_data.Data
        # classes = prep_data.classes
        self.classes = np.load(test_main.params.main_param_path + '/classes.npy')
        self.Filenames = prep_data.Filenames
        self.class_weights_tensor = torch.load(test_main.params.main_param_path + '/class_weights_tensor.pt')

        if train_main.params.ttkind == 'mixed':
            self.trainFilenames = Data[0]
            trainXimage = Data[1]
            trainXfeat = Data[9]
            trY = Data[2]
            trX = [trainXimage, trainXfeat]
        elif train_main.params.ttkind == 'image' and train_main.params.compute_extrafeat == 'yes':
            self.trainFilenames = Data[0]
            trainXimage = Data[1]
            trY = Data[2]
            trainXfeat = Data[9]
            trX = [trainXimage, trainXfeat]
        elif train_main.params.ttkind == 'feat':
            self.trainFilenames = Data[0]
            trY = Data[2]
            trainXfeat = Data[9]
            trX = [trainXfeat]
        else:
            self.trainFilenames = Data[0]
            trX = Data[1]
            trY = Data[2]

        data_train = trX.astype(np.float64)
        data_train = 255 * data_train
        self.X_train = data_train.astype(np.uint8)
        # self.y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))], dtype=object)
        self.y_train = np.concatenate([np.where(self.classes == uid) if np.where(self.classes == uid) else print(
            'The folder should match the trained classes') for uid in trY]).ravel()

        return

    def create_data_loaders(self, train_main):
        # self.checkpoint_path = test_main.params.model_path

        test_dataset = CreateDataset(X=self.X_train)
        self.test_dataloader = DataLoader(test_dataset, 32, shuffle=False, num_workers=4,
                                          pin_memory=True)

    def create_data_loaders_with_y(self, test_main):
        # self.checkpoint_path = test_main.params.model_path

        test_dataset = CreateDataset_with_y(X=self.X_train, y=self.y_train, TTA_type=test_main.params.TTA_type)
        self.test_dataloader = DataLoader(test_dataset, 32, shuffle=False, num_workers=4,
                                          pin_memory=True)
        # torch.save(test_dataset, test_main.params.main_param_path + '/test_dataloader.pt')
        # DATA = [self.X_train, self.y_train]
        # with open(test_main.params.main_param_path + '/test_data.pickle', 'wb') as a:
        #     pickle.dump(DATA, a, protocol=4)


class CreateDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, X):
        """Initialization"""
        self.X = X

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        image = self.X[index]
        X = self.transform(image)
        sample = X
        return sample

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor()])
    transform_y = T.Compose([T.ToTensor()])


class CreateDataset_with_y(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, X, y, TTA_type):
        """Initialization"""
        self.X = X
        self.y = y
        self.TTA_type = TTA_type

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        image = self.X[index]
        label = self.y[index]
        TTA_type = self.TTA_type
        if TTA_type == 0:
            X = self.transform(image)
        elif TTA_type == 1:
            X = self.transform_TTA_1(image)
        elif TTA_type == 2:
            X = self.transform_TTA_2(image)
        elif TTA_type == 3:
            X = self.transform_TTA_3(image)
        elif TTA_type == 4:
            X = self.transform_TTA_4(image)
        elif TTA_type == 5:
            X = self.transform_TTA_5(image)
        elif TTA_type == 6:
            X = self.transform_TTA_6(image)
        elif TTA_type == 7:
            X = self.transform_TTA_7(image)
        elif TTA_type == 8:
            X = self.transform_TTA_8(image)
        elif TTA_type == 9:
            X = self.transform_TTA_9(image)
        elif TTA_type == 10:
            X = self.transform_TTA_10(image)
        elif TTA_type == 11:
            X = self.transform_TTA_11(image)
        elif TTA_type == 12:
            X = self.transform_TTA_12(image)
        elif TTA_type == 13:
            X = self.transform_TTA_13(image)
        elif TTA_type == 14:
            X = self.transform_TTA_14(image)
        elif TTA_type == 15:
            X = self.transform_TTA_15(image)
        elif TTA_type == 16:
            X = self.transform_TTA_16(image)
        elif TTA_type == 17:
            X = self.transform_TTA_17(image)
        elif TTA_type == 18:
            X = self.transform_TTA_18(image)
        elif TTA_type == 19:
            X = self.transform_TTA_19(image)
        elif TTA_type == 20:
            X = self.transform_TTA_20(image)
        elif TTA_type == 21:
            X = self.transform_TTA_21(image)
        elif TTA_type == 22:
            X = self.transform_TTA_22(image)
        elif TTA_type == 23:
            X = self.transform_TTA_23(image)
        y = label
        sample = [X, y]
        return sample

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor()])
    transform_y = T.Compose([T.ToTensor()])

    ## Test-time augmentations
    # Rotate 90
    transform_TTA_1 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(90, 90)),
        T.ToTensor()])
    transform_TTA_1_y = T.Compose([T.ToTensor()])

    # Rotate 180
    transform_TTA_2 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(180, 180)),
        T.ToTensor()])
    transform_TTA_2_y = T.Compose([T.ToTensor()])

    # Rotate 270
    transform_TTA_3 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(270, 270)),
        T.ToTensor()])
    transform_TTA_3_y = T.Compose([T.ToTensor()])

    # Horizontal flip
    transform_TTA_4 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomHorizontalFlip(p=1),
        T.ToTensor()])
    transform_TTA_4_y = T.Compose([T.ToTensor()])

    # Random rotation
    transform_TTA_5 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(0, 360)),
        T.ToTensor()])
    transform_TTA_5_y = T.Compose([T.ToTensor()])

    # Flip + rotation
    transform_TTA_6 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomHorizontalFlip(p=1),
        T.RandomRotation(degrees=(0, 360)),
        T.ToTensor()])
    transform_TTA_6_y = T.Compose([T.ToTensor()])

    # Flip + rotate 90
    transform_TTA_7 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomHorizontalFlip(p=1),
        T.RandomRotation(degrees=(90, 90)),
        T.ToTensor()])
    transform_TTA_7_y = T.Compose([T.ToTensor()])

    # Flip + rotate 180
    transform_TTA_8 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomHorizontalFlip(p=1),
        T.RandomRotation(degrees=(180, 180)),
        T.ToTensor()])
    transform_TTA_8_y = T.Compose([T.ToTensor()])

    # Flip + rotate 270
    transform_TTA_9 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomHorizontalFlip(p=1),
        T.RandomRotation(degrees=(270, 270)),
        T.ToTensor()])
    transform_TTA_9_y = T.Compose([T.ToTensor()])

    # Rotate 45
    transform_TTA_10 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(45, 45)),
        T.ToTensor()])
    transform_TTA_10_y = T.Compose([T.ToTensor()])

    # Rotate 135
    transform_TTA_11 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(135, 135)),
        T.ToTensor()])
    transform_TTA_11_y = T.Compose([T.ToTensor()])

    # Rotate 225
    transform_TTA_12 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(225, 225)),
        T.ToTensor()])
    transform_TTA_12_y = T.Compose([T.ToTensor()])

    # Rotate 315
    transform_TTA_13 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(315, 315)),
        T.ToTensor()])
    transform_TTA_13_y = T.Compose([T.ToTensor()])

    # Rotate 30
    transform_TTA_14 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(30, 30)),
        T.ToTensor()])
    transform_TTA_14_y = T.Compose([T.ToTensor()])

    # Rotate 60
    transform_TTA_15 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(60, 60)),
        T.ToTensor()])
    transform_TTA_15_y = T.Compose([T.ToTensor()])

    # Rotate 120
    transform_TTA_16 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(120, 120)),
        T.ToTensor()])
    transform_TTA_16_y = T.Compose([T.ToTensor()])

    # Rotate 150
    transform_TTA_17 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(150, 150)),
        T.ToTensor()])
    transform_TTA_17_y = T.Compose([T.ToTensor()])

    # Rotate 210
    transform_TTA_18 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(210, 210)),
        T.ToTensor()])
    transform_TTA_18_y = T.Compose([T.ToTensor()])

    # Rotate 240
    transform_TTA_19 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(240, 240)),
        T.ToTensor()])
    transform_TTA_19_y = T.Compose([T.ToTensor()])

    # Rotate 300
    transform_TTA_20 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(300, 300)),
        T.ToTensor()])
    transform_TTA_20_y = T.Compose([T.ToTensor()])

    # Rotate 330
    transform_TTA_21 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(330, 330)),
        T.ToTensor()])
    transform_TTA_21_y = T.Compose([T.ToTensor()])

    # Vertical flip
    transform_TTA_22 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomVerticalFlip(p=1),
        T.ToTensor()])
    transform_TTA_22_y = T.Compose([T.ToTensor()])

    # Horizontal + Vertical flip
    transform_TTA_23 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomHorizontalFlip(p=1),
        T.RandomVerticalFlip(p=1),
        T.ToTensor()])
    transform_TTA_23_y = T.Compose([T.ToTensor()])