import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import ray
from ray import tune, air
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch

from hyperopt import hp

import timm

import argparse


class Net(nn.Module):
    def __init__(self, dropout_1=0.4, dropout_2=0.3, fc_node=512, n_classes=35, base_model='tf_efficientnet_b7', HP_last_layer=True, add_layer='yes'):
        super(Net, self).__init__()
        basemodel = timm.create_model(base_model, pretrained=True, num_classes=n_classes)
        
        # additional layers
        if add_layer == 'yes':
            if base_model == 'deit_base_distilled_patch16_224':
                in_features = basemodel.get_classifier()[-1].in_features
                pretrained_layers = list(basemodel.children())[:-2]
                additional_layers = nn.Sequential(
                                        nn.Dropout(p=dropout_1),
                                        nn.Linear(in_features=in_features, out_features=fc_node),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=dropout_2),
                                        nn.Linear(in_features=fc_node, out_features=n_classes),
                                        )
                self.model = nn.Sequential(*pretrained_layers, additional_layers)

            else:
                in_features = basemodel.get_classifier().in_features
                pretrained_layers = list(basemodel.children())[:-1]
                additional_layers = nn.Sequential(
                                        nn.Dropout(p=dropout_1),
                                        nn.Linear(in_features=in_features, out_features=fc_node),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=dropout_2),
                                        nn.Linear(in_features=fc_node, out_features=n_classes),
                                        )
                self.model = nn.Sequential(*pretrained_layers, additional_layers)

            if HP_last_layer == True:
                n_layer = 0
                for param in self.model.parameters():
                    n_layer += 1
                    param.requires_grad = False

                for i, param in enumerate(self.model.parameters()):
                    if i + 1 > n_layer - 5: 
                        param.requires_grad = True

            else:
                for param in self.model.parameters():
                    param.requires_grad = True
        
        elif add_layer == 'no':
            self.model = basemodel

            if HP_last_layer == True:
                n_layer = 0
                for param in self.model.parameters():
                    n_layer += 1
                    param.requires_grad = False

                for i, param in enumerate(self.model.parameters()):
                    if i + 1 > n_layer - 2: 
                        param.requires_grad = True

            else:
                for param in self.model.parameters():
                    param.requires_grad = True
        
    def forward(self, x):
        x = self.model(x)
        return x


def train(model, optimizer, criterion, train_loader, device, base_model, add_layer):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        if add_layer == 'yes':
            if base_model == 'deit_base_distilled_patch16_224' or base_model == 'vit_base_patch16_224' or base_model == 'vit_base_patch16_224.mae':
                output = torch.mean(output, 1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, device, base_model, add_layer):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0

    targets = []
    outputs = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            if add_layer == 'yes':
                if base_model == 'deit_base_distilled_patch16_224' or base_model == 'vit_base_patch16_224' or base_model == 'vit_base_patch16_224.mae':
                    output = torch.mean(output, 1)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            outputs.append(output)
            targets.append(target)

        outputs = torch.cat(outputs)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.argmax(outputs, axis=1)

        targets = torch.cat(targets)
        targets = targets.cpu().detach().numpy()

        acc = correct / total
        f1 = f1_score(outputs, targets, average='macro')

    return acc, f1


class AugmentedDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, X, y, aug_type):
        """Initialization"""
        self.X = X
        self.y = y
        self.aug_type = aug_type

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        image = self.X[index]
        label = self.y[index]
        aug_type = self.aug_type
        if aug_type == 'high':
            X = self.transform1(image)
        elif aug_type == 'medium':
            X = self.transform2(image)
        elif aug_type == 'low':
            X = self.transform3(image)
        else:
            X = self.transform4(image)
        y = label
        sample = [X, y]
        return sample

    transform1 = T.Compose([
            T.ToPILImage(),
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomResizedCrop(size=(224, 224), scale=(0.35, 1.0), ratio=(0.9, 1.1)),
            # T.ElasticTransform(alpha=1.0, sigma=1.0),
            T.RandomPerspective(distortion_scale=0.5, p=0.2),
            T.RandomPosterize(bits=5, p=0.2),
            T.RandomSolarize(threshold=128, p=0.2),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            T.RandomAutocontrast(p=0.5),
            T.ColorJitter(brightness=(0.7, 1.3), contrast=(0.8, 1.2), saturation=(0.5, 1.0), hue=(-0.01, 0.01)),
            T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            T.RandomRotation(degrees=(0, 360)),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.0), shear=(1, 1, 1, 1)),
            # T.RandAugment(),
            # T.TrivialAugmentWide(),
            # T.AugMix(),
            # T.RandomErasing(),
            # T.Grayscale(),
            # T.RandomInvert(),
            # T.RandomEqualize(),
            T.ToTensor()])
    transform1_y = T.Compose([T.ToTensor()])
    
    transform2 = T.Compose([
            T.ToPILImage(),
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomResizedCrop(size=(224, 224), scale=(0.35, 1.0), ratio=(0.9, 1.1)),
            # T.ElasticTransform(alpha=1.0, sigma=1.0),
            T.RandomPerspective(distortion_scale=0.5, p=0.2),
            T.ColorJitter(brightness=(0.7, 1.3), contrast=(0.8, 1.2), saturation=(0.5, 1.0), hue=(-0.01, 0.01)),
            T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            T.RandomRotation(degrees=(0, 360)),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.0), shear=(1, 1, 1, 1)),
            T.ToTensor()])
    transform2_y = T.Compose([T.ToTensor()])

    transform3 = T.Compose([
            T.ToPILImage(),
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=(0, 180)),
            T.ToTensor()])
    transform3_y = T.Compose([T.ToTensor()])

    transform4 = T.Compose([
            T.ToPILImage(),
            T.Resize(224),
            T.ToTensor()])
    transform4_y = T.Compose([T.ToTensor()])


class CreateDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.X[index]
        label = self.y[index]
        X = self.transform(image)
        y = label
        #         y = self.transform_y(label)
        #         sample = {'image': X, 'label': label}
        sample = [X, y]
        return sample

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor()])    
    transform_y = T.Compose([T.ToTensor()])


def trainable(config):
    # Data Setup
    Data = pd.read_pickle(config['data'])
    classes = np.load(config['classes'])
    n_classes = len(classes)
    base_model = config['base_model']
    HP_last_layer = config['last_layer']
    add_layer = config['add_layer']
    batch_size = config['batch_size']
    
    trainFilenames = Data[0]
    trX = Data[1]
    trY = Data[2]
    testFilenames = Data[3]
    teX = Data[4]
    teY = Data[5]
    valFilenames = Data[6]
    veX = Data[7]
    veY = Data[8]
    
    classes_int = np.unique(np.argmax(trY, axis=1))
    
    y_train_max = trY.argmax(axis=1)  # The class that the classifier would bet on
    y_train = np.array([classes_int[y_train_max[i]] for i in range(len(y_train_max))], dtype=object)

    y_test_max = teY.argmax(axis=1)  # The class that the classifier would bet on
    y_test = np.array([classes_int[y_test_max[i]] for i in range(len(y_test_max))], dtype=object)

    y_val_max = veY.argmax(axis=1)  # The class that the classifier would bet on
    y_val = np.array([classes_int[y_val_max[i]] for i in range(len(y_val_max))], dtype=object)
    
    data_train = trX.astype(np.float64)
    data_train = 255 * data_train
    X_train = data_train.astype(np.uint8)

    data_test = teX.astype(np.float64)
    data_test = 255 * data_test
    X_test = data_test.astype(np.uint8)

    data_val = veX.astype(np.float64)
    data_val = 255 * data_val
    X_val = data_val.astype(np.uint8)

    
    
    train_dataset = AugmentedDataset(X=X_train,y=y_train,aug_type=config['aug_type'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = CreateDataset(X=X_val,y=y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if add_layer == 'yes':
        model = Net(config['dropout_1'], config['dropout_2'], int(config['fc_node']), n_classes, base_model, HP_last_layer, add_layer)
    elif add_layer == 'no':
        model = Net(n_classes=n_classes, base_model=base_model, HP_last_layer=HP_last_layer, add_layer=add_layer)
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    if HP_last_layer == True:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        
    criterion = nn.CrossEntropyLoss()
    
    criterion.cuda(0)
    
    for i in range(config['n_epochs']):
        train(model, optimizer, criterion, train_loader, device, base_model, add_layer)
        acc, f1 = test(model, val_loader, device, base_model, add_layer)

        # Send the current training result back to Tune
        # tune.report(mean_accuracy=acc)
        tune.report(mean_f1=f1)


        # if i % 5 == 0:
        #     # This saves the model to the trial directory
        #     torch.save(model.state_dict(), "./model.pth")
    


    for param in model.parameters():
        param.requires_grad = True

    optimizer_full = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=config["weight_decay"])
    for i in range(20):
        train(model, optimizer_full, criterion, train_loader, device, base_model, add_layer)
        acc, f1 = test(model, val_loader, device, base_model, add_layer)

        tune.report(mean_f1=f1)


parser = argparse.ArgumentParser(description='Hyperparameter tuning')
parser.add_argument('-data_pickle', default='/home/EAWAG/chenchen/out/train_out/Zooplankton/deit_20221209_batch16/Data.pickle')
parser.add_argument('-class_npy', default='/home/EAWAG/chenchen/out/train_out/Zooplankton/deit_20221209_batch16/classes.npy')
parser.add_argument('-base_model', type=str, default='vit_base_patch16_224.mae')
parser.add_argument('-outpath', default='/home/EAWAG/chenchen/out/HP_out/efficientnetb7_20221222/')
parser.add_argument('-multinode', choices=['yes', 'no'], default='yes')
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-last_layer', type=bool, default=True)
parser.add_argument('-add_layer', choices=['yes', 'no'], default='yes')
parser.add_argument('-aug_type', choices=['high', 'medium', 'low'], default='low')
parser.add_argument('-n_samples', type=int, default=50)
parser.add_argument('-gpu_per_trial', type=float, default=1)
parser.add_argument('-cpu_per_trial', type=float, default=0)
parser.add_argument('-epoch', type=int, default=10)
parser.add_argument('-patience', type=int, default=3)
parser.add_argument('-bayes', choices=['yes', 'no'], default='yes')
parser.add_argument('-bayes_init_search', type=int, default=5)
parser.add_argument('-lr_low', type=float, default=1e-6)
parser.add_argument('-lr_up', type=float, default=1e-3)
parser.add_argument('-weight_decay_low', type=float, default=1e-2)
parser.add_argument('-weight_decay_up', type=float, default=5e-2)
parser.add_argument('-dropout_low', type=float, default=0.0)
parser.add_argument('-dropout_up', type=float, default=1.0)
parser.add_argument('-fc_node_low', type=int, default=128)
parser.add_argument('-fc_node_up', type=int, default=2048)
args = parser.parse_args()


if __name__ == "__main__":
    if args.add_layer == 'yes':
        config = {
                "data": args.data_pickle,
                "classes": args.class_npy,
                "base_model": args.base_model,
                "n_epochs": args.epoch,
                "last_layer": args.last_layer,
                "add_layer": args.add_layer,
                "aug_type": args.aug_type,
                "batch_size": args.batch_size,
                "lr": tune.loguniform(args.lr_low, args.lr_up),
                "weight_decay": tune.uniform(args.weight_decay_low, args.weight_decay_up),
                "dropout_1": tune.quniform(args.dropout_low, args.dropout_up, 0.1),
                "dropout_2": tune.quniform(args.dropout_low, args.dropout_up, 0.1),
                "fc_node": tune.uniform(args.fc_node_low, args.fc_node_up)
                }

    elif args.add_layer == 'no':
        config = {
                "data": args.data_pickle,
                "classes": args.class_npy,
                "base_model": args.base_model,
                "n_epochs": args.epoch,
                "last_layer": args.last_layer,
                "add_layer": args.add_layer,
                "aug_type": args.aug_type,
                "batch_size": args.batch_size,
                "lr": tune.loguniform(args.lr_low, args.lr_up),
                "weight_decay": tune.uniform(args.weight_decay_low, args.weight_decay_up)
                }

    out_dir = args.outpath

    # scheduler = ASHAScheduler(metric="mean_accuracy", mode="max", grace_period=args.patience)
    # scheduler = ASHAScheduler(metric="mean_f1", mode="max", grace_period=args.patience)

    reporter = CLIReporter(max_report_frequency=30)

    trainable_with_resources = tune.with_resources(trainable, {"gpu": args.gpu_per_trial, "cpu": args.cpu_per_trial})

    if args.multinode == 'yes':
        ray.init('auto')

    if args.bayes == 'yes':
        # bayesopt = BayesOptSearch(metric="mean_accuracy", mode="max", random_search_steps=args.bayes_init_search)
        bayesopt = BayesOptSearch(metric="mean_f1", mode="max", random_search_steps=args.bayes_init_search)

        tuner = tune.Tuner(
            trainable_with_resources,
            tune_config=tune.TuneConfig(
                search_alg=bayesopt,
                # scheduler=scheduler,
                num_samples=args.n_samples
            ),
            param_space=config,
            run_config=air.RunConfig(local_dir=out_dir, name="ray_tune", progress_reporter=reporter)
        )

    else:
        tuner = tune.Tuner(
            trainable_with_resources,
            tune_config=tune.TuneConfig(
                # scheduler=scheduler,
                num_samples=args.n_samples
            ),
            param_space=config,
            run_config=air.RunConfig(local_dir=out_dir, name="ray_tune", progress_reporter=reporter)
        )

    results = tuner.fit()

    # best_config = results.get_best_result(metric='mean_accuracy', mode='max', scope='all').config
    # df_overview_best = results.get_dataframe(filter_metric='mean_accuracy', filter_mode='max')
    # df_overview_final = results.get_dataframe()

    best_config = results.get_best_result(metric='mean_f1', mode='max', scope='all').config
    df_overview_best = results.get_dataframe(filter_metric='mean_f1', filter_mode='max')
    df_overview_final = results.get_dataframe()

    f = open(out_dir + '/best_config.txt', 'w')
    f.write(str(best_config))
    f.close()
    df_overview_best.to_excel(out_dir + '/HP_report_best.xlsx', header=True, index=True)
    df_overview_final.to_excel(out_dir + '/HP_report_final.xlsx', header=True, index=True)

