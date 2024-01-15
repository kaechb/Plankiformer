import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from distinctipy import distinctipy
import argparse
import random
from pathlib import Path
from scipy.stats import sem, pearsonr
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, QuantileRegressor


random.seed(100)
plt.rcParams['figure.dpi'] = 300
# plt.rcParams.update({'font.size': 15})
# plt.rc('axes', labelsize=15)
# plt.rc('legend', fontsize=15)
# plt.rc('figure', titlesize=15) 


def preprocess_distance_txt(distance_txt_path):
    distance_txt = pd.read_csv(distance_txt_path, header=None)
    distance = distance_txt.drop(distance_txt.index[-1])
    df_distance = pd.DataFrame(columns=['class', 'distance'], index=range(len(distance)))

    for i in range(len(distance)):
        df_distance['class'][i] = distance.iloc[i].item()[0:20].strip()
        df_distance['distance'][i] = float(distance.iloc[i].item()[20:28])

    return df_distance

def preprocess_F1_txt(F1_txt_path):
    F1_txt = pd.read_csv(F1_txt_path, header=None)
    F1 = F1_txt.drop(F1_txt.index[0:6])
    F1 = F1.reset_index(drop=True)
    F1 = F1.drop(F1.index[-3:])
    df_F1 = pd.DataFrame(columns=['class', 'precision', 'recall', 'F1', 'support'], index=range(len(F1)))

    for i in range(len(F1)):
        df_F1['class'][i] = F1.iloc[i].item()[0:20].strip()
        df_F1['precision'][i] = float(F1.iloc[i].item()[24:33])
        df_F1['recall'][i] = float(F1.iloc[i].item()[34:43])
        df_F1['F1'][i] = float(F1.iloc[i].item()[44:53])
        df_F1['support'][i] = int(F1.iloc[i].item()[54:])

    return df_F1

def dataframe_distance_f1(testsets, distance_txt_paths, F1_txt_paths, threshold):
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = preprocess_distance_txt(distance_txt_paths[i])
        df_F1 = preprocess_F1_txt(F1_txt_paths[i])
        if threshold > 0:
            df_F1 = df_F1.drop(df_F1[df_F1['support'] < threshold].index)
            df_F1 = df_F1.reset_index(drop=True)
        df_distance_F1 = pd.DataFrame(columns=['class', 'testset', 'distance', 'precision', 'recall', 'F1'], index=range(len(df_distance)))
        df_distance_F1['class'] = df_distance['class']
        df_distance_F1['testset'] = testsets[i]
        df_distance_F1['distance'] = df_distance['distance']
        df_distance_F1['precision'] = df_F1['precision']
        df_distance_F1['recall'] = df_F1['recall']
        df_distance_F1['F1'] = df_F1['F1']
        df_distance_F1['support'] = df_F1['support']
        df = pd.concat([df, df_distance_F1])
    df = df.reset_index(drop=True)

    return df

def dataframe_distance_f1_drop(testsets, distance_txt_paths, train_F1_txt_path, F1_txt_paths, threshold):
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = preprocess_distance_txt(distance_txt_paths[i])
        df_F1 = preprocess_F1_txt(F1_txt_paths[i])
        df_F1_train = preprocess_F1_txt(train_F1_txt_path)
        if threshold > 0:
            df_F1 = df_F1.drop(df_F1[df_F1['support'] < threshold].index)
            df_F1 = df_F1.reset_index(drop=True)
            df_F1_train = df_F1_train.drop(df_F1_train[df_F1_train['support'] < threshold].index)
            df_F1_train = df_F1_train.reset_index(drop=True)
        df_distance_F1 = pd.DataFrame(columns=['class','testset', 'precision', 'recall', 'F1'], index=range(len(df_distance)))
        df_distance_F1['class'] = df_distance['class']
        df_distance_F1['testset'] = testsets[i]
        df_distance_F1['precision'] = df_F1['precision']
        df_distance_F1['recall'] = df_F1['recall']
        df_distance_F1['F1'] = df_F1['F1']
        dict_test = {key:F1 for key, F1 in zip(df_F1['class'], df_F1['F1'])}
        dict_train = {key:F1 for key, F1 in zip(df_F1_train['class'], df_F1_train['F1'])}
        dict_test_recall = {key:recall for key, recall in zip(df_F1['class'], df_F1['recall'])}
        dict_train_recall = {key:recall for key, recall in zip(df_F1_train['class'], df_F1_train['recall'])}
        F1_drop = []
        for i in dict_test.keys():
            F1_drop_class = 1 - np.divide(dict_test[i], dict_train[i])
            F1_drop.append(F1_drop_class)
        df_distance_F1['F1_drop'] = F1_drop
        recall_drop = []
        for i in dict_test_recall.keys():
            recall_drop_class = 1 - np.divide(dict_test_recall[i], dict_train_recall[i])
            recall_drop.append(recall_drop_class)
        df_distance_F1['recall_drop'] = recall_drop
        
        df_distance_F1['distance'] = df_distance['distance']
        df = pd.concat([df, df_distance_F1])
    df = df.reset_index(drop=True)

    return df

def dataframe_distance_f1_per_feature(testsets, distance_dfs, train_F1_txt_path, F1_txt_paths, threshold):
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = distance_dfs[i]

        df_distance.columns = df_distance.columns.str.replace(r'^m(\d{2})$', r'$M\1$', regex=True)
        df_distance.columns = df_distance.columns.str.replace(r'^mu(\d{2})$', r'$μ\1$', regex=True)
        df_distance.columns = df_distance.columns.str.replace(r'^nu(\d{2})$', r'$η\1$', regex=True)
        df_distance.columns = df_distance.columns.str.replace(r'^hu(\d{1})$', r'$I\1$', regex=True)

        df_distance = df_distance.reset_index(level=0)
        df_distance.columns.values[0] = 'class'
        df_F1 = preprocess_F1_txt(F1_txt_paths[i])
        df_F1_train = preprocess_F1_txt(train_F1_txt_path)
        if threshold > 0:
            df_F1 = df_F1.drop(df_F1[df_F1['support'] < threshold].index)
            df_F1 = df_F1.reset_index(drop=True)
            df_F1_train = df_F1_train.drop(df_F1_train[df_F1_train['support'] < threshold].index)
            df_F1_train = df_F1_train.reset_index(drop=True)
        df_distance_F1 = pd.DataFrame(columns=['class','testset', 'precision', 'recall', 'F1'], index=range(len(df_distance)))
        df_distance_F1['class'] = df_distance['class']
        df_distance_F1['testset'] = testsets[i]
        df_distance_F1['precision'] = df_F1['precision']
        df_distance_F1['recall'] = df_F1['recall']
        df_distance_F1['F1'] = df_F1['F1']
        dict_test = {key:F1 for key, F1 in zip(df_F1['class'], df_F1['F1'])}
        dict_train = {key:F1 for key, F1 in zip(df_F1_train['class'], df_F1_train['F1'])}
        F1_drop = []
        for i in dict_test.keys():
            F1_drop_class = 1 - np.divide(dict_test[i], dict_train[i])
            F1_drop.append(F1_drop_class)
        df_distance_F1['F1_drop'] = F1_drop
        for idistance in df_distance.columns.values[1:]:
            df_distance_F1[idistance] = df_distance[idistance]
        # df_distance_F1['distance'] = df_distance['distance']
        df = pd.concat([df, df_distance_F1])
    df = df.reset_index(drop=True)
    df['mean_distance'] = df.iloc[:, 6:].mean(axis=1)

    return df

def plot_distance_F1_scatter(testsets, distance_txt_paths, F1_txt_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type):
    
    df = dataframe_distance_f1(testsets, distance_txt_paths, F1_txt_paths, threshold)

    # F1
    plt.figure(figsize=(10, 10))
    plt.suptitle(model)
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set ' + '(' + feature_or_pixel + ')')
    plt.ylabel('F1-score')
    # colors = cm.nipy_spectral(np.linspace(0, 1, len(np.unique(df['class']))))
    random.seed(100)
    colors = distinctipy.get_colors(len(np.unique(df['class'])), pastel_factor=0.7)
    for iclass, c in zip(np.unique(df['class']), colors):
        plt.scatter(df[df['class'] == iclass].distance, df[df['class'] == iclass].F1, label=iclass, c=np.array([c]))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)

    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    # recall
    plt.figure(figsize=(10, 10))
    plt.suptitle(model)
    plt.subplot(1, 1, 1)
    plt.xlabel('Distance to training set ' + '(' + feature_or_pixel + ')')
    plt.ylabel('Recall')
    # colors = cm.nipy_spectral(np.linspace(0, 1, len(np.unique(df['class']))))
    random.seed(100)
    colors = distinctipy.get_colors(len(np.unique(df['class'])), pastel_factor=0.7)
    for iclass, c in zip(np.unique(df['class']), colors):
        plt.scatter(df[df['class'] == iclass].distance, df[df['class'] == iclass].recall, label=iclass, c=np.array([c]))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance_F1_err(testsets, distance_txt_paths, F1_txt_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type):
    
    df = dataframe_distance_f1(testsets, distance_txt_paths, F1_txt_paths, threshold)

    F1_mean = []
    F1_std = []
    F1_sem = []
    recall_mean = []
    recall_std = []
    recall_sem = []
    distance_mean = []
    distance_std = []
    distance_sem = []

    for i in testsets:
        F1_mean_testset = np.average(df[df['testset'] == i]['F1'])
        F1_mean.append(F1_mean_testset)
        F1_std_testset = np.std(df[df['testset'] == i]['F1'])
        F1_std.append(F1_std_testset)
        F1_sem_testset = sem(df[df['testset'] == i]['F1'])
        F1_sem.append(F1_sem_testset)

        recall_mean_testset = np.average(df[df['testset'] == i]['recall'])
        recall_mean.append(recall_mean_testset)
        recall_std_testset = np.std(df[df['testset'] == i]['recall'])
        recall_std.append(recall_std_testset)
        recall_sem_testset = sem(df[df['testset'] == i]['recall'])
        recall_sem.append(recall_sem_testset)

        distance_mean_testset = np.average(df[df['testset'] == i]['distance'])
        distance_mean.append(distance_mean_testset)
        distance_std_testset = np.std(df[df['testset'] == i]['distance'])
        distance_std.append(distance_std_testset)
        distance_sem_testset = sem(df[df['testset'] == i]['distance'])
        distance_sem.append(distance_sem_testset)


    # F1
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('Testset')
    ax.set_ylabel('F1-score')
    x = range(len(testsets))
    plt.xticks(x, labels=testsets)
    plt.ylim(-0.1, 1.1)

    plt.errorbar(x, F1_mean, yerr=F1_sem, fmt='-s', color='g', capsize=5, label='F1')

    ax_2 = ax.twinx()
    ax_2.set_ylabel('1 - ' + distance_type + ' Distance to training set')
    plt.errorbar(x, np.subtract(1, distance_mean), yerr=distance_sem, fmt='-s', color='r', capsize=5, label='1 - ' + distance_type+' Distance')
    # plt.errorbar(x, distance_mean, yerr=distance_sem, fmt='-s', color='r', capsize=5, label=distance_type+' Distance')
    plt.ylim(0.95*np.min(np.subtract(1, distance_mean)), 1.05*np.max(np.subtract(1, distance_mean)))
    # plt.ylim(0.4, 1)

    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()
    ax.clear()

    # recall
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('Testset')
    ax.set_ylabel('Recall')
    x = range(len(testsets))
    plt.xticks(x, labels=testsets)
    plt.ylim(-0.1, 1.1)

    plt.errorbar(x, recall_mean, yerr=recall_sem, fmt='-s', color='g', capsize=5, label='recall')

    ax_2 = ax.twinx()
    ax_2.set_ylabel('1 - ' + distance_type + ' Distance to training set')
    plt.errorbar(x, np.subtract(1, distance_mean), yerr=distance_sem, fmt='-s', color='r', capsize=5, label='1 - ' + distance_type +' Distance')
    # plt.errorbar(x, distance_mean, yerr=distance_sem, fmt='-s', color='r', capsize=5, label=distance_type+' Distance')
    plt.ylim(0.95*np.min(np.subtract(1, distance_mean)), 1.05*np.max(np.subtract(1, distance_mean)))
    # plt.ylim(0.4, 1)

    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()
    ax.clear()

def plot_distance_F1_testset(testsets, distance_txt_paths, F1_txt_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting global correlation of distance and F1 (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    
    df = dataframe_distance_f1(testsets, distance_txt_paths, F1_txt_paths, threshold)

    if class_filter == 'yes':
        df_ID = df[df['testset'] == 'ID_test']
        dropped_classes_1 = df_ID[df_ID['F1'] < 0.75]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        for iclass in dropped_classes:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)
        # n_datapoint = df['testset'].value_counts()
        # dropped_testsets = n_datapoint[n_datapoint < 2].index.to_list()
        # for itestset in dropped_testsets:
        #     testsets.remove(itestset)
        #     df = df.drop(df[df['testset'] == itestset].index)
        #     df = df.reset_index(drop=True)

    F1_mean = []
    F1_sem = []
    # recall_mean = []
    # recall_sem = []
    distance_mean = []
    distance_sem = []
    for i in testsets:
        F1_mean_testset = np.average(df[df['testset'] == i]['F1'])
        F1_mean.append(F1_mean_testset)
        if len(df[df['testset'] == i]['F1']) > 1:
            F1_sem_testset = sem(df[df['testset'] == i]['F1'])
        else:
            F1_sem_testset = 0
        F1_sem.append(F1_sem_testset)
        # recall_mean_testset = np.average(df[df['testset'] == i]['recall'])
        # recall_mean.append(recall_mean_testset)
        # if len(df[df['testset'] == i]['recall']) > 1:
        #     recall_sem_testset = sem(df[df['testset'] == i]['recall'])
        # else:
        #     recall_sem_testset = 0
        # recall_sem.append(recall_sem_testset)
        distance_mean_testset = np.average(df[df['testset'] == i]['distance'])
        distance_mean.append(distance_mean_testset)
        if len(df[df['testset'] == i]['distance']) > 1:
            distance_sem_testset = sem(df[df['testset'] == i]['distance'])
        else:
            distance_sem_testset = 0
        distance_sem.append(distance_sem_testset)

    # F1
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1-score')
    random.seed(100)
    colors = distinctipy.get_colors(len(testsets), pastel_factor=0.7)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=F1_mean[i], xerr=distance_sem[i], yerr=F1_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))

    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')

    correlation, p_value = pearsonr(np.array(distance_mean), np.array(F1_mean))
    reg = LinearRegression().fit(np.array(distance_mean).reshape(-1, 1), np.array(F1_mean))
    y_fit = reg.predict(np.array(distance_mean).reshape(-1, 1))
    plt.plot(np.array(distance_mean), y_fit, color='red', label='regression line')
    # plt.title('r = %s' % round(correlation, 3))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_testset_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_testset_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1-score')
    x = np.array([])
    y = np.array([])
    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')
    #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].F1_drop, label=itestset, alpha=0.6)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=F1_mean[i], xerr=distance_sem[i], yerr=F1_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].F1, label=itestset, alpha=0.6, c=np.array([c]))
        x = np.concatenate((x, np.array(df[df['testset'] == itestset]['distance'])), axis=None)
        y = np.concatenate((y, np.array(df[df['testset'] == itestset].F1)), axis=None)
    correlation, p_value = pearsonr(x, y)
    # reg = LinearRegression().fit(x.reshape(-1, 1), y)
    # y_fit = reg.predict(x.reshape(-1, 1))
    x, y = x.astype(float), y.astype(float)
    xy = np.stack((x, y), axis=0)
    xy_sorted = xy.T[xy.T[:, 0].argsort()].T
    x, y = xy_sorted[0], xy_sorted[1]
    a, b = np.polyfit(x, y, deg=1)
    y_fit = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    plt.fill_between(x, y_fit - y_err, y_fit + y_err, alpha=0.2)
    # plt.plot(x, y_fit - y_err)
    # plt.plot(x, y_fit + y_err)
    plt.plot(x, y_fit, color='red', label='Mean regression')

    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
        y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
        predictions[quantile] = y_pred
    for quantile, y_pred in predictions.items():
        plt.plot(x.reshape(-1, 1), y_pred, label=f"Quantile: {quantile}", dashes=[6, 2])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.title('r = %s' % round(correlation, 3))
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1-score')
    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')
    #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].F1_drop, label=itestset, alpha=0.6)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=F1_mean[i], xerr=distance_sem[i], yerr=F1_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].F1, label=itestset, alpha=0.6, c=np.array([c]))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()




    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 1, 1)
    # plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    # plt.ylabel('F1-score')
    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=F1_mean[i], xerr=distance_sem[i], yerr=F1_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset)
    # plt.legend(loc='best')
    # plt.tight_layout()  
    # Path(outpath).mkdir(parents=True, exist_ok=True)
    # if PCA == 'yes':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_testset_threshold_' + str(threshold) + '.png', dpi=300)
    # elif PCA == 'no':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_testset_threshold_' + str(threshold) + '.png', dpi=300)
    # plt.close()

    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 1, 1)
    # plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    # plt.ylabel('F1-score')
    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=F1_mean[i], xerr=distance_sem[i], yerr=F1_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset)
    #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].F1, label=itestset)
    # plt.legend(loc='best')
    # plt.tight_layout()  
    # Path(outpath).mkdir(parents=True, exist_ok=True)
    # if PCA == 'yes':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    # elif PCA == 'no':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    # plt.close()

    # # recall
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 1, 1)
    # plt.xlabel('Distance to training set ' + '(' + feature_or_pixel + ')')
    # plt.ylabel('Recall')
    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=recall_mean[i], xerr=distance_sem[i], yerr=recall_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # Path(outpath).mkdir(parents=True, exist_ok=True)
    # if PCA == 'yes':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_testset_threshold_' + str(threshold) + '.png', dpi=300)
    # elif PCA == 'no':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_testset_threshold_' + str(threshold) + '.png', dpi=300)
    # plt.close()

    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 1, 1)
    # plt.xlabel('Distance to training set ' + '(' + feature_or_pixel + ')')
    # plt.ylabel('Recall')
    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=recall_mean[i], xerr=distance_sem[i], yerr=recall_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset)
    #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].recall, label=itestset)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # Path(outpath).mkdir(parents=True, exist_ok=True)
    # if PCA == 'yes':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    # elif PCA == 'no':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    # plt.close()



    # plt.figure(figsize=(12, 12))
    # plt.subplot(1, 1, 1)
    # plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    # plt.ylabel('Recall')
    # random.seed(100)
    # colors = distinctipy.get_colors(len(testsets), pastel_factor=0.7)
    # for i, (itestset, c) in enumerate(zip(testsets, colors)):
    #     plt.errorbar(x=distance_mean[i], y=recall_mean[i], xerr=distance_sem[i], yerr=recall_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))

    # # for i, itestset in enumerate(testsets):
    # #     plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')

    # correlation, p_value = pearsonr(np.array(distance_mean), np.array(recall_mean))
    # reg = LinearRegression().fit(np.array(distance_mean).reshape(-1, 1), np.array(recall_mean))
    # y_fit = reg.predict(np.array(distance_mean).reshape(-1, 1))
    # plt.plot(np.array(distance_mean), y_fit, color='red', label='regression line')
    # # plt.title('r = %s' % round(correlation, 3))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.tight_layout()
    # Path(outpath).mkdir(parents=True, exist_ok=True)
    # if PCA == 'yes':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_testset_err_threshold_' + str(threshold) + '.png', dpi=300)
    # elif PCA == 'no':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_testset_err_threshold_' + str(threshold) + '.png', dpi=300)
    # plt.close()

    # plt.figure(figsize=(12, 12))
    # plt.subplot(1, 1, 1)
    # plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    # plt.ylabel('Recall')
    # x = np.array([])
    # y = np.array([])
    # # for i, itestset in enumerate(testsets):
    # #     plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')
    # #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].recall_drop, label=itestset, alpha=0.6)
    # for i, (itestset, c) in enumerate(zip(testsets, colors)):
    #     plt.errorbar(x=distance_mean[i], y=recall_mean[i], xerr=distance_sem[i], yerr=recall_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
    #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].recall, label=itestset, alpha=0.6, c=np.array([c]))
    #     x = np.concatenate((x, np.array(df[df['testset'] == itestset]['distance'])), axis=None)
    #     y = np.concatenate((y, np.array(df[df['testset'] == itestset].recall)), axis=None)
    # correlation, p_value = pearsonr(x, y)
    # # reg = LinearRegression().fit(x.reshape(-1, 1), y)
    # # y_fit = reg.predict(x.reshape(-1, 1))
    # x, y = x.astype(float), y.astype(float)
    # xy = np.stack((x, y), axis=0)
    # xy_sorted = xy.T[xy.T[:, 0].argsort()].T
    # x, y = xy_sorted[0], xy_sorted[1]
    # a, b = np.polyfit(x, y, deg=1)
    # y_fit = a * x + b
    # y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    # plt.fill_between(x, y_fit - y_err, y_fit + y_err, alpha=0.2)
    # # plt.plot(x, y_fit - y_err)
    # # plt.plot(x, y_fit + y_err)
    # plt.plot(x, y_fit, color='red', label='Mean regression')

    # quantiles = [0.05, 0.5, 0.95]
    # predictions = {}
    # for quantile in quantiles:
    #     qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
    #     y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
    #     predictions[quantile] = y_pred
    # for quantile, y_pred in predictions.items():
    #     plt.plot(x.reshape(-1, 1), y_pred, label=f"Quantile: {quantile}", dashes=[6, 2])

    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # # plt.title('r = %s' % round(correlation, 3))
    # plt.tight_layout()
    # if PCA == 'yes':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    # elif PCA == 'no':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    # plt.close()

    # plt.figure(figsize=(12, 12))
    # plt.subplot(1, 1, 1)
    # plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    # plt.ylabel('Recall')
    # # for i, itestset in enumerate(testsets):
    # #     plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')
    # #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].recall_drop, label=itestset, alpha=0.6)
    # for i, (itestset, c) in enumerate(zip(testsets, colors)):
    #     plt.errorbar(x=distance_mean[i], y=recall_mean[i], xerr=distance_sem[i], yerr=recall_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
    #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].recall, label=itestset, alpha=0.6, c=np.array([c]))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.tight_layout()
    # if PCA == 'yes':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    # elif PCA == 'no':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    # plt.close()

def plot_distance_F1_class(testsets, distance_txt_paths, F1_txt_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting global correlation of distance and F1 (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    
    df = dataframe_distance_f1(testsets, distance_txt_paths, F1_txt_paths, threshold)

    if class_filter == 'yes':
        df_ID = df[df['testset'] == 'ID_test']
        dropped_classes_1 = df_ID[df_ID['F1'] < 0.75]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        for iclass in dropped_classes:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)
        # n_datapoint = df['testset'].value_counts()
        # dropped_testsets = n_datapoint[n_datapoint < 2].index.to_list()
        # for itestset in dropped_testsets:
        #     testsets.remove(itestset)
        #     df = df.drop(df[df['testset'] == itestset].index)
        #     df = df.reset_index(drop=True)
    
    # elif class_filter == 'no':
    #     n_testset = df['class'].value_counts()
    #     dropped_classes = n_testset[n_testset < 2].index.to_list()
    #     for iclass in dropped_classes:
    #         df = df.drop(df[df['class'] == iclass].index)
    #         df = df.reset_index(drop=True)

    classes = np.unique(df['class']).tolist()

    F1_mean = []
    F1_sem = []
    # recall_mean = []
    # recall_sem = []
    distance_mean = []
    distance_sem = []
    for i in classes:
        F1_mean_class = np.average(df[df['class'] == i]['F1'])
        F1_mean.append(F1_mean_class)
        if len(df[df['class'] == i]['F1']) > 1:
            F1_sem_class = sem(df[df['class'] == i]['F1'])
        else:
            F1_sem_class = 0
        F1_sem.append(F1_sem_class)
        # recall_mean_class = np.average(df[df['class'] == i]['recall'])
        # recall_mean.append(recall_mean_class)
        # if len(df[df['class'] == i]['recall']) > 1:
        #     recall_sem_class = sem(df[df['class'] == i]['recall'])
        # else:
        #     recall_sem_class = 0
        # recall_sem.append(recall_sem_class)
        distance_mean_class = np.average(df[df['class'] == i]['distance'])
        distance_mean.append(distance_mean_class)
        if len(df[df['class'] == i]['distance']) > 1:
            distance_sem_class = sem(df[df['class'] == i]['distance'])
        else:
            distance_sem_class = 0
        distance_sem.append(distance_sem_class)

    # F1
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1-score')
    random.seed(100)
    colors = distinctipy.get_colors(len(classes), pastel_factor=0.7)
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=F1_mean[i], xerr=distance_sem[i], yerr=F1_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))
    
    correlation, p_value = pearsonr(np.array(distance_mean), np.array(F1_mean))
    reg = LinearRegression().fit(np.array(distance_mean).reshape(-1, 1), np.array(F1_mean))
    y_fit = reg.predict(np.array(distance_mean).reshape(-1, 1))
    plt.plot(np.array(distance_mean), y_fit, color='red', label='regression line')
    # plt.title('r = %s' % round(correlation, 3))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()
    
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.tight_layout()
    # Path(outpath).mkdir(parents=True, exist_ok=True)
    # if PCA == 'yes':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_class_threshold_' + str(threshold) + '.png', dpi=300)
    # elif PCA == 'no':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_class_threshold_' + str(threshold) + '.png', dpi=300)
    # plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1-score')
    x = np.array([])
    y = np.array([])

    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=F1_mean[i], xerr=distance_sem[i], yerr=F1_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].F1, label=iclass, alpha=0.6, c=np.array([c]))
        x = np.concatenate((x, np.array(df[df['class'] == iclass]['distance'])), axis=None)
        y = np.concatenate((y, np.array(df[df['class'] == iclass].F1)), axis=None)
    correlation, p_value = pearsonr(x, y)
    # reg = LinearRegression().fit(x.reshape(-1, 1), y)
    # y_fit = reg.predict(x.reshape(-1, 1))
    x, y = x.astype(float), y.astype(float)
    xy = np.stack((x, y), axis=0)
    xy_sorted = xy.T[xy.T[:, 0].argsort()].T
    x, y = xy_sorted[0], xy_sorted[1]
    a, b = np.polyfit(x, y, deg=1)
    y_fit = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    plt.fill_between(x, y_fit - y_err, y_fit + y_err, alpha=0.2)
    # plt.plot(x, y_fit - y_err)
    # plt.plot(x, y_fit + y_err)
    plt.plot(x, y_fit, color='red', label='Mean regression')

    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
        y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
        predictions[quantile] = y_pred
    for quantile, y_pred in predictions.items():
        plt.plot(x.reshape(-1, 1), y_pred, label=f"Quantile: {quantile}", dashes=[6, 2])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.title('r = %s' % round(correlation, 3))
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_class_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_class_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1-score')
    
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=F1_mean[i], xerr=distance_sem[i], yerr=F1_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].F1, label=iclass, alpha=0.6, c=np.array([c]))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_class_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_class_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1-score')
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].F1, label=iclass, alpha=0.6, c=np.array([c]))
        x = np.array([])
        y = np.array([])
        for itestset in df[df['class'] == iclass].testset.values:
            x = np.concatenate((x, df[(df['class'] == iclass) & (df['testset'] == itestset)].distance), axis=None)
            y = np.concatenate((y, df[(df['class'] == iclass) & (df['testset'] == itestset)].F1), axis=None)
        # correlation, p_value = pearsonr(x, y)
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        y_fit = reg.predict(x.reshape(-1, 1))
        plt.plot(x, y_fit, color=np.array([c]))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_class_scatter_per_class_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_class_scatter_per_class_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()





    # # recall
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 1, 1)
    # plt.xlabel('Distance to training set ' + '(' + feature_or_pixel + ')')
    # plt.ylabel('Recall')
    # for i, iclass in enumerate(classes):
    #     plt.errorbar(x=distance_mean[i], y=recall_mean[i], xerr=distance_sem[i], yerr=recall_sem[i], fmt='s', capsize=3, label=iclass)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.tight_layout()
    # Path(outpath).mkdir(parents=True, exist_ok=True)
    # if PCA == 'yes':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_class_threshold_' + str(threshold) + '.png', dpi=300)
    # elif PCA == 'no':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_class_threshold_' + str(threshold) + '.png', dpi=300)
    # plt.close()

def plot_distance_F1_drop_testset(testsets, distance_txt_paths, train_F1_txt_path, F1_txt_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):    
    print('------------plotting global correlation of distance and F1 drop (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    
    df = dataframe_distance_f1_drop(testsets, distance_txt_paths, train_F1_txt_path, F1_txt_paths, threshold)
    
    if class_filter == 'yes':
        df_ID = df[df['testset'] == 'ID_test']
        dropped_classes_1 = df_ID[df_ID['F1'] < 0.75]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        for iclass in dropped_classes:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)
        # n_datapoint = df['testset'].value_counts()
        # dropped_testsets = n_datapoint[n_datapoint < 2].index.to_list()
        # for itestset in dropped_testsets:
        #     testsets.remove(itestset)
        #     df = df.drop(df[df['testset'] == itestset].index)
        #     df = df.reset_index(drop=True)

    F1_drop_mean = []
    F1_drop_sem = []
    recall_drop_mean = []
    recall_drop_sem = []
    distance_mean = []
    distance_sem = []
    for i in testsets:
        F1_drop_mean_testset = np.average(df[df['testset'] == i]['F1_drop'])
        F1_drop_mean.append(F1_drop_mean_testset)
        if len(df[df['testset'] == i]['F1_drop']) > 1:
            F1_drop_sem_testset = sem(df[df['testset'] == i]['F1_drop'])
        else:
            F1_drop_sem_testset = 0
        F1_drop_sem.append(F1_drop_sem_testset)

        recall_drop_mean_testset = np.average(df[df['testset'] == i]['recall_drop'])
        recall_drop_mean.append(recall_drop_mean_testset)
        if len(df[df['testset'] == i]['recall_drop']) > 1:
            recall_drop_sem_testset = sem(df[df['testset'] == i]['recall_drop'])
        else:
            recall_drop_sem_testset = 0
        recall_drop_sem.append(recall_drop_sem_testset)

        distance_mean_testset = np.average(df[df['testset'] == i].distance)
        distance_mean.append(distance_mean_testset)
        if len(df[df['testset'] == i].distance) > 1:
            distance_sem_testset = sem(df[df['testset'] == i].distance)
        else:
            distance_sem_testset = 0
        distance_sem.append(distance_sem_testset)

    # F1
    plt.figure(figsize=(9, 9))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1 drop ratio')
    random.seed(100)
    colors = distinctipy.get_colors(len(testsets), pastel_factor=0.7)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))

    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')

    correlation, p_value = pearsonr(np.array(distance_mean), np.array(F1_drop_mean))
    reg = LinearRegression().fit(np.array(distance_mean).reshape(-1, 1), np.array(F1_drop_mean))
    y_fit = reg.predict(np.array(distance_mean).reshape(-1, 1))
    plt.plot(np.array(distance_mean), y_fit, color='red', label='regression line')
    # plt.title('r = %s' % round(correlation, 3))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_drop_testset_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_drop_testset_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(9, 10))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$', fontsize=20)
    plt.ylabel('F1 drop ratio', fontsize=20)
    x = np.array([])
    y = np.array([])
    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')
    #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].F1_drop, label=itestset, alpha=0.6)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].F1_drop, label=itestset, alpha=0.6, c=np.array([c]))
        x = np.concatenate((x, np.array(df[df['testset'] == itestset]['distance'])), axis=None)
        y = np.concatenate((y, np.array(df[df['testset'] == itestset].F1_drop)), axis=None)
    correlation, p_value = pearsonr(x, y)
    # reg = LinearRegression().fit(x.reshape(-1, 1), y)
    # y_fit = reg.predict(x.reshape(-1, 1))
    x, y = x.astype(float), y.astype(float)
    xy = np.stack((x, y), axis=0)
    xy_sorted = xy.T[xy.T[:, 0].argsort()].T
    x, y = xy_sorted[0], xy_sorted[1]
    a, b = np.polyfit(x, y, deg=1)
    y_fit = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    plt.fill_between(x, y_fit - y_err, y_fit + y_err, alpha=0.2)
    # plt.plot(x, y_fit - y_err)
    # plt.plot(x, y_fit + y_err)
    plt.plot(x, y_fit, color='red', label='Mean regression')

    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
        y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
        predictions[quantile] = y_pred
    for quantile, y_pred in predictions.items():
        plt.plot(x.reshape(-1, 1), y_pred, label=f"Quantile: {quantile}", dashes=[6, 2])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, ncol=5)
    # plt.title('r = %s' % round(correlation, 3))
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_drop_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_drop_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1 drop ratio')
    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')
    #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].F1_drop, label=itestset, alpha=0.6)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].F1_drop, label=itestset, alpha=0.6, c=np.array([c]))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_drop_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_drop_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()


    # recall
    plt.figure(figsize=(9, 9))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('Recall drop ratio')
    random.seed(100)
    colors = distinctipy.get_colors(len(testsets), pastel_factor=0.7)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))

    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')

    correlation, p_value = pearsonr(np.array(distance_mean), np.array(recall_drop_mean))
    reg = LinearRegression().fit(np.array(distance_mean).reshape(-1, 1), np.array(recall_drop_mean))
    y_fit = reg.predict(np.array(distance_mean).reshape(-1, 1))
    plt.plot(np.array(distance_mean), y_fit, color='red', label='regression line')
    # plt.title('r = %s' % round(correlation, 3))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_drop_testset_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_drop_testset_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(9, 10))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$', fontsize=20)
    plt.ylabel('Recall drop ratio', fontsize=20)
    x = np.array([])
    y = np.array([])
    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')
    #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].recall_drop, label=itestset, alpha=0.6)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].recall_drop, label=itestset, alpha=0.6, c=np.array([c]))
        x = np.concatenate((x, np.array(df[df['testset'] == itestset]['distance'])), axis=None)
        y = np.concatenate((y, np.array(df[df['testset'] == itestset].recall_drop)), axis=None)
    correlation, p_value = pearsonr(x, y)
    # reg = LinearRegression().fit(x.reshape(-1, 1), y)
    # y_fit = reg.predict(x.reshape(-1, 1))
    x, y = x.astype(float), y.astype(float)
    xy = np.stack((x, y), axis=0)
    xy_sorted = xy.T[xy.T[:, 0].argsort()].T
    x, y = xy_sorted[0], xy_sorted[1]
    a, b = np.polyfit(x, y, deg=1)
    y_fit = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    plt.fill_between(x, y_fit - y_err, y_fit + y_err, alpha=0.2)
    # plt.plot(x, y_fit - y_err)
    # plt.plot(x, y_fit + y_err)
    plt.plot(x, y_fit, color='red', label='Mean regression')

    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
        y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
        predictions[quantile] = y_pred
    for quantile, y_pred in predictions.items():
        plt.plot(x.reshape(-1, 1), y_pred, label=f"Quantile: {quantile}", dashes=[6, 2])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, ncol=5)
    # plt.title('r = %s' % round(correlation, 3))
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_drop_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_drop_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('Recall drop ratio')
    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')
    #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].recall_drop, label=itestset, alpha=0.6)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].recall_drop, label=itestset, alpha=0.6, c=np.array([c]))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_drop_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_drop_testset_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()



    plt.figure(figsize=(17, 9))
    plt.subplots_adjust(left=0.05, bottom=0.15, right=0.98, top=0.98, wspace=0.1)
    plt.subplot(1, 2, 1)
    plt.xlabel('KL divergence ' + r'$D_\mathcal{KL}$', fontsize=20)
    plt.ylabel('F1 drop ratio', fontsize=20)
    x = np.array([])
    y = np.array([])
    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')
    #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].F1_drop, label=itestset, alpha=0.6)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].F1_drop, label=itestset, alpha=0.6, c=np.array([c]))
        x = np.concatenate((x, np.array(df[df['testset'] == itestset]['distance'])), axis=None)
        y = np.concatenate((y, np.array(df[df['testset'] == itestset].F1_drop)), axis=None)
    correlation, p_value = pearsonr(x, y)
    # reg = LinearRegression().fit(x.reshape(-1, 1), y)
    # y_fit = reg.predict(x.reshape(-1, 1))
    x, y = x.astype(float), y.astype(float)
    xy = np.stack((x, y), axis=0)
    xy_sorted = xy.T[xy.T[:, 0].argsort()].T
    x, y = xy_sorted[0], xy_sorted[1]
    a, b = np.polyfit(x, y, deg=1)
    y_fit = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    plt.fill_between(x, y_fit - y_err, y_fit + y_err, alpha=0.2)
    # plt.plot(x, y_fit - y_err)
    # plt.plot(x, y_fit + y_err)
    plt.plot(x, y_fit, color='red', label='Mean regression')

    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
        y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
        predictions[quantile] = y_pred
    for quantile, y_pred in predictions.items():
        plt.plot(x.reshape(-1, 1), y_pred, label=f"Quantile: {quantile}", dashes=[6, 2])

    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, ncol=5)
    # plt.title('r = %s' % round(correlation, 3))
    # plt.tight_layout()

    plt.subplot(1, 2, 2)
    plt.xlabel('KL divergence ' + r'$D_\mathcal{KL}$', fontsize=20)
    plt.ylabel('Recall drop ratio', fontsize=20)
    x = np.array([])
    y = np.array([])
    # for i, itestset in enumerate(testsets):
    #     plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)')
    #     plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].recall_drop, label=itestset, alpha=0.6)
    for i, (itestset, c) in enumerate(zip(testsets, colors)):
        plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=itestset + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['testset'] == itestset].distance, y=df[df['testset'] == itestset].recall_drop, label=itestset, alpha=0.6, c=np.array([c]))
        x = np.concatenate((x, np.array(df[df['testset'] == itestset]['distance'])), axis=None)
        y = np.concatenate((y, np.array(df[df['testset'] == itestset].recall_drop)), axis=None)
    correlation, p_value = pearsonr(x, y)
    # reg = LinearRegression().fit(x.reshape(-1, 1), y)
    # y_fit = reg.predict(x.reshape(-1, 1))
    x, y = x.astype(float), y.astype(float)
    xy = np.stack((x, y), axis=0)
    xy_sorted = xy.T[xy.T[:, 0].argsort()].T
    x, y = xy_sorted[0], xy_sorted[1]
    a, b = np.polyfit(x, y, deg=1)
    y_fit = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    plt.fill_between(x, y_fit - y_err, y_fit + y_err, alpha=0.2)
    # plt.plot(x, y_fit - y_err)
    # plt.plot(x, y_fit + y_err)
    plt.plot(x, y_fit, color='red', label='Mean regression')

    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
        y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
        predictions[quantile] = y_pred
    for quantile, y_pred in predictions.items():
        plt.plot(x.reshape(-1, 1), y_pred, label=f"Quantile: {quantile}", dashes=[6, 2])

    plt.legend(loc='upper center', bbox_to_anchor=(-0.09, -0.07), fancybox=True, ncol=10)
    # plt.title('r = %s' % round(correlation, 3))
    # plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_recall_drop_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_recall_drop_testset_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance_F1_drop_class(testsets, distance_txt_paths, train_F1_txt_path, F1_txt_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):    
    print('------------plotting global correlation of distance and F1 drop (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    
    df = dataframe_distance_f1_drop(testsets, distance_txt_paths, train_F1_txt_path, F1_txt_paths, threshold)
    
    if class_filter == 'yes':
        df_ID = df[df['testset'] == 'ID_test']
        ## remove junk classes
        # dropped_classes_1 = df_ID[df_ID['F1'] < 0.75]['class'].to_list()
        ## remain junk classes
        dropped_classes_1 = df_ID[df_ID['F1'] < 0]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        for iclass in dropped_classes:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)
        # n_datapoint = df['testset'].value_counts()
        # dropped_testsets = n_datapoint[n_datapoint < 2].index.to_list()
        # for itestset in dropped_testsets:
        #     testsets.remove(itestset)
        #     df = df.drop(df[df['testset'] == itestset].index)
        #     df = df.reset_index(drop=True)

    # elif class_filter == 'no':
    #     n_testset = df['class'].value_counts()
    #     dropped_classes = n_testset[n_testset < 2].index.to_list()
    #     for iclass in dropped_classes:
    #         df = df.drop(df[df['class'] == iclass].index)
    #         df = df.reset_index(drop=True)

    classes = np.unique(df['class']).tolist()
    
    F1_drop_mean = []
    F1_drop_sem = []
    recall_drop_mean = []
    recall_drop_sem = []
    distance_mean = []
    distance_sem = []
    for i in classes:
        F1_drop_mean_class = np.average(df[df['class'] == i]['F1_drop'])
        F1_drop_mean.append(F1_drop_mean_class)
        if len(df[df['class'] == i]['F1_drop']) > 1:
            F1_drop_sem_class = sem(df[df['class'] == i]['F1_drop'])
        else:
            F1_drop_sem_class = 0
        F1_drop_sem.append(F1_drop_sem_class)

        recall_drop_mean_class = np.average(df[df['class'] == i]['recall_drop'])
        recall_drop_mean.append(recall_drop_mean_class)
        if len(df[df['class'] == i]['recall_drop']) > 1:
            recall_drop_sem_class = sem(df[df['class'] == i]['recall_drop'])
        else:
            recall_drop_sem_class = 0
        recall_drop_sem.append(recall_drop_sem_class)

        distance_mean_class = np.average(df[df['class'] == i].distance)
        distance_mean.append(distance_mean_class)
        if len(df[df['class'] == i].distance) > 1:
            distance_sem_class = sem(df[df['class'] == i].distance)
        else:
            distance_sem_class = 0
        distance_sem.append(distance_sem_class)

    # F1
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1 drop ratio')
    random.seed(100)
    colors = distinctipy.get_colors(len(classes), pastel_factor=0.7)
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))

    correlation, p_value = pearsonr(np.array(distance_mean), np.array(F1_drop_mean))
    reg = LinearRegression().fit(np.array(distance_mean).reshape(-1, 1), np.array(F1_drop_mean))
    y_fit = reg.predict(np.array(distance_mean).reshape(-1, 1))
    plt.plot(np.array(distance_mean), y_fit, color='red', label='regression line')
    # plt.title('r = %s' % round(correlation, 3))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_drop_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_drop_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.tight_layout()
    # if PCA == 'yes':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_drop_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    # elif PCA == 'no':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_drop_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    # plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1 drop ratio')
    x = np.array([])
    y = np.array([])
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].F1_drop, label=iclass, alpha=0.6, c=np.array([c]))
        x = np.concatenate((x, np.array(df[df['class'] == iclass]['distance'])), axis=None)
        y = np.concatenate((y, np.array(df[df['class'] == iclass].F1_drop)), axis=None)
    correlation, p_value = pearsonr(x, y)
    # reg = LinearRegression().fit(x.reshape(-1, 1), y)
    # y_fit = reg.predict(x.reshape(-1, 1))
    x, y = x.astype(float), y.astype(float)
    xy = np.stack((x, y), axis=0)
    xy_sorted = xy.T[xy.T[:, 0].argsort()].T
    x, y = xy_sorted[0], xy_sorted[1]
    a, b = np.polyfit(x, y, deg=1)
    y_fit = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    plt.fill_between(x, y_fit - y_err, y_fit + y_err, alpha=0.2)
    # plt.plot(x, y_fit - y_err)
    # plt.plot(x, y_fit + y_err)
    plt.plot(x, y_fit, color='red', label='Mean regression')

    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
        y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
        predictions[quantile] = y_pred
    for quantile, y_pred in predictions.items():
        plt.plot(x.reshape(-1, 1), y_pred, label=f"Quantile: {quantile}", dashes=[6, 2])
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.title('r = %s' % round(correlation, 3))
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_drop_class_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_drop_class_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1 drop ratio')
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=F1_drop_mean[i], xerr=distance_sem[i], yerr=F1_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].F1_drop, label=iclass, alpha=0.6, c=np.array([c]))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_drop_class_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_drop_class_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()


    slope_per_class = {}
    plt.figure(figsize=(9, 9))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$', fontsize=20)
    plt.ylabel('F1 drop ratio', fontsize=20)
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].F1_drop, label=iclass, alpha=0.6, c=np.array([c]))
        x = np.array([])
        y = np.array([])
        for itestset in df[df['class'] == iclass].testset.values:
            x = np.concatenate((x, df[(df['class'] == iclass) & (df['testset'] == itestset)].distance), axis=None)
            y = np.concatenate((y, df[(df['class'] == iclass) & (df['testset'] == itestset)].F1_drop), axis=None)
        # correlation, p_value = pearsonr(x, y)
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        y_fit = reg.predict(x.reshape(-1, 1))
        slope_per_class[iclass] = reg.coef_[0]
        plt.plot(x, y_fit, color=np.array([c]))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_drop_class_scatter_per_class_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_drop_class_scatter_per_class_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    sorted_slope_class = dict(sorted(slope_per_class.items(), key=lambda item: item[1], reverse=True))
    plt.figure(figsize=(9, 7))
    plt.subplot(1, 1, 1)
    plt.bar(x=range(len(sorted_slope_class)), height=sorted_slope_class.values(), color='forestgreen')
    plt.xticks(range(len(sorted_slope_class)), labels=sorted_slope_class.keys(), rotation=45, rotation_mode='anchor', ha='right', fontsize=15)
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('Sensitivity ' + r'$Q_{F_1}$', fontsize=20)
    plt.yticks(fontsize=15)
    plt.ylim((-0.5, 4))
    plt.grid(alpha=0.5, axis='y', which='both')
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_class_slope_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_class_slope_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()


    # recall
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('Recall drop ratio')
    random.seed(100)
    colors = distinctipy.get_colors(len(classes), pastel_factor=0.7)
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))

    correlation, p_value = pearsonr(np.array(distance_mean), np.array(recall_drop_mean))
    reg = LinearRegression().fit(np.array(distance_mean).reshape(-1, 1), np.array(recall_drop_mean))
    y_fit = reg.predict(np.array(distance_mean).reshape(-1, 1))
    plt.plot(np.array(distance_mean), y_fit, color='red', label='regression line')
    # plt.title('r = %s' % round(correlation, 3))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_drop_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_drop_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.tight_layout()
    # if PCA == 'yes':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_drop_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    # elif PCA == 'no':
    #     plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_drop_class_err_threshold_' + str(threshold) + '.png', dpi=300)
    # plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('Recall drop ratio')
    x = np.array([])
    y = np.array([])
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].recall_drop, label=iclass, alpha=0.6, c=np.array([c]))
        x = np.concatenate((x, np.array(df[df['class'] == iclass]['distance'])), axis=None)
        y = np.concatenate((y, np.array(df[df['class'] == iclass].recall_drop)), axis=None)
    correlation, p_value = pearsonr(x, y)
    # reg = LinearRegression().fit(x.reshape(-1, 1), y)
    # y_fit = reg.predict(x.reshape(-1, 1))
    x, y = x.astype(float), y.astype(float)
    xy = np.stack((x, y), axis=0)
    xy_sorted = xy.T[xy.T[:, 0].argsort()].T
    x, y = xy_sorted[0], xy_sorted[1]
    a, b = np.polyfit(x, y, deg=1)
    y_fit = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    plt.fill_between(x, y_fit - y_err, y_fit + y_err, alpha=0.2)
    # plt.plot(x, y_fit - y_err)
    # plt.plot(x, y_fit + y_err)
    plt.plot(x, y_fit, color='red', label='Mean regression')

    quantiles = [0.05, 0.5, 0.95]
    predictions = {}
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
        y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1, 1))
        predictions[quantile] = y_pred
    for quantile, y_pred in predictions.items():
        plt.plot(x.reshape(-1, 1), y_pred, label=f"Quantile: {quantile}", dashes=[6, 2])
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    # plt.title('r = %s' % round(correlation, 3))
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_drop_class_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_drop_class_scatter_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('Recall drop ratio')
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.errorbar(x=distance_mean[i], y=recall_drop_mean[i], xerr=distance_sem[i], yerr=recall_drop_sem[i], fmt='s', elinewidth=1, capthick=1, capsize=3, markersize=12, label=iclass + ' (mean)', c=np.array([c]))
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].recall_drop, label=iclass, alpha=0.6, c=np.array([c]))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_drop_class_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_drop_class_scatter_err_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('Recall drop ratio')
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        plt.scatter(x=df[df['class'] == iclass].distance, y=df[df['class'] == iclass].recall_drop, label=iclass, alpha=0.6, c=np.array([c]))
        x = np.array([])
        y = np.array([])
        for itestset in df[df['class'] == iclass].testset.values:
            x = np.concatenate((x, df[(df['class'] == iclass) & (df['testset'] == itestset)].distance), axis=None)
            y = np.concatenate((y, df[(df['class'] == iclass) & (df['testset'] == itestset)].recall_drop), axis=None)
        # correlation, p_value = pearsonr(x, y)
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        y_fit = reg.predict(x.reshape(-1, 1))
        plt.plot(x, y_fit, color=np.array([c]))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_recall_drop_class_scatter_per_class_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_recall_drop_class_scatter_per_class_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance_F1_drop(testsets, distance_dfs, train_F1_txt_path, F1_txt_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting correlation of distance and F1 drop per component (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    
    df = dataframe_distance_f1_per_feature(testsets, distance_dfs, train_F1_txt_path, F1_txt_paths, threshold)

    if class_filter == 'yes':
        df_ID = df[df['testset'] == 'ID_test']
        dropped_classes_1 = df_ID[df_ID['F1'] < 0.75]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        for iclass in dropped_classes:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)
        n_datapoint = df['testset'].value_counts()
        dropped_testsets = n_datapoint[n_datapoint < 2].index.to_list()
        for itestset in dropped_testsets:
            testsets.remove(itestset)
            df = df.drop(df[df['testset'] == itestset].index)
            df = df.reset_index(drop=True)
    
    plt.figure(figsize=(9, 8))
    # plt.figure(figsize=(12, 9))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}^{f}$', fontsize=20)
    plt.ylabel('F1 drop ratio', fontsize=20)
    random.seed(100)
    colors = distinctipy.get_colors(len(df.columns.values[6:-1]), pastel_factor=0.7)
    F1_drop_mean_global = []
    F1_drop_sem_global = []
    distance_mean_global = []
    distance_sem_global = []
    
    slope_per_feature = {}
    for ifeature, c in zip(df.columns.values[6:-1], colors):
        F1_drop_mean = []
        F1_drop_sem = []
        distance_mean = []
        distance_sem = []
        for i, itestset in enumerate(testsets):
            F1_drop_mean_testset = np.average(df[df['testset'] == itestset]['F1_drop'])
            F1_drop_mean.append(F1_drop_mean_testset)
            F1_drop_mean_global.append(F1_drop_mean_testset)
            F1_drop_sem_testset = sem(df[df['testset'] == itestset]['F1_drop'])
            F1_drop_sem.append(F1_drop_sem_testset)
            F1_drop_sem_global.append(F1_drop_sem_testset)

            distance_mean_testset = np.average(df[df['testset'] == itestset][ifeature])
            distance_mean.append(distance_mean_testset)
            distance_mean_global.append(distance_mean_testset)
            distance_sem_testset = sem(df[df['testset'] == itestset][ifeature])
            distance_sem.append(distance_sem_testset)
            distance_sem_global.append(distance_sem_testset)
        
        weights = 1 / np.sqrt(np.array(distance_sem)**2 + np.array(F1_drop_sem)**2)
        params, covariance = curve_fit(linear_func, distance_mean, F1_drop_mean, sigma=weights)
        a = params[0]
        slope_per_feature[ifeature] = a
        b = params[1]
        F1_drop_mean_pred = a * np.array(distance_mean) + b

#         # All features
#         plt.errorbar(distance_mean, F1_drop_mean, xerr=distance_sem, yerr=F1_drop_sem, fmt='s', elinewidth=1, capthick=1, capsize=3, c=np.array([c]), alpha=0.3)
#         # plt.errorbar(distance_mean, F1_drop_mean, fmt='s', elinewidth=1, capthick=1, capsize=3, c=np.array([c]), alpha=0.3)
# #         plt.plot(distance_mean, F1_drop_mean_pred, color=np.array([c]), label=ifeature)
#         if a > 0.7:
#             plt.axline((distance_mean[0], F1_drop_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=3)
#         else:
#             plt.axline((distance_mean[0], F1_drop_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=0.8)
#         # plt.axline((distance_mean[0], F1_drop_mean_pred[0]), (distance_mean[-1], F1_drop_mean_pred[-1]), color=np.array([c]), label=ifeature)


        # Simplified
        if a > 0.7:
            plt.axline((distance_mean[0], F1_drop_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=3)
        

    weights_global = 1 / np.sqrt(np.array(distance_sem_global)**2 + np.array(F1_drop_sem_global)**2)
    params_global, covariance_global = curve_fit(linear_func, distance_mean_global, F1_drop_mean_global, sigma=weights_global)
    a_global = params_global[0]
    b_global = params_global[1]
    F1_drop_mean_pred_global = a_global * np.array(distance_mean_global) + b_global
    plt.axline((distance_mean_global[0], F1_drop_mean_pred_global[0]), slope=a_global, color='black', label='average', linewidth=3, linestyle='--')

    plt.ylim(-0.03, 0.3)
    plt.xlim(0.05, 0.4)

    plt.legend(bbox_to_anchor=(1, 1), fancybox=True, ncol=1)
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_drop_per_feature_threshold_' + str(threshold) + '_simplified.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_drop_per_feature_threshold_' + str(threshold) + '_simplified.png', dpi=300)
    plt.close()

    sorted_slope = dict(sorted(slope_per_feature.items(), key=lambda item: item[1], reverse=True))
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 1, 1)
    plt.bar(x=range(len(df.columns.values[6:-1])), height=sorted_slope.values(), color='chocolate')
    plt.xticks(range(len(df.columns.values[6:-1])), labels=sorted_slope.keys(), rotation=45, rotation_mode='anchor', ha='right')
    plt.xlabel('Feature', fontsize=35)
    plt.ylabel('Sensitivity ' + r'$Q_{F_1}$', fontsize=35)
    plt.yticks(fontsize=20)
    plt.grid(alpha=0.5, axis='y', which='both')
    plt.ylim((-0.4, 1.4))
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_feature_slope_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_feature_slope_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance_F1(testsets, distance_dfs, train_F1_txt_path, F1_txt_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting correlation of distance and F1 per component (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    
    df = dataframe_distance_f1_per_feature(testsets, distance_dfs, train_F1_txt_path, F1_txt_paths, threshold)

    if class_filter == 'yes':
        df_ID = df[df['testset'] == 'ID_test']
        dropped_classes_1 = df_ID[df_ID['F1'] < 0.75]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        for iclass in dropped_classes:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)
        n_datapoint = df['testset'].value_counts()
        dropped_testsets = n_datapoint[n_datapoint < 2].index.to_list()
        for itestset in dropped_testsets:
            testsets.remove(itestset)
            df = df.drop(df[df['testset'] == itestset].index)
            df = df.reset_index(drop=True)
    
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}^{f}$')
    plt.ylabel('F1-score')
    random.seed(100)
    colors = distinctipy.get_colors(len(df.columns.values[6:-1]), pastel_factor=0.7)
    for ifeature, c in zip(df.columns.values[6:-1], colors):
        F1_mean = []
        F1_sem = []
        distance_mean = []
        distance_sem = []
        for i, itestset in enumerate(testsets):
            F1_mean_testset = np.average(df[df['testset'] == itestset]['F1'])
            F1_mean.append(F1_mean_testset)
            F1_sem_testset = sem(df[df['testset'] == itestset]['F1'])
            F1_sem.append(F1_sem_testset)

            distance_mean_testset = np.nanmean(df[df['testset'] == itestset][ifeature])
            distance_mean.append(distance_mean_testset)
            distance_sem_testset = sem(df[df['testset'] == itestset][ifeature])
            distance_sem.append(distance_sem_testset)
        
        weights = 1 / np.sqrt(np.array(distance_sem)**2 + np.array(F1_sem)**2)
        params, covariance = curve_fit(linear_func, distance_mean, F1_mean, sigma=weights)
        a = params[0]
        b = params[1]
        F1_mean_pred = a * np.array(distance_mean) + b
        plt.errorbar(distance_mean, F1_mean, xerr=distance_sem, yerr=F1_sem, fmt='s', elinewidth=1, capthick=1, capsize=3, c=np.array([c]), alpha=0.3)
        # plt.errorbar(distance_mean, F1_mean, fmt='s', elinewidth=1, capthick=1, capsize=3, c=np.array([c]), alpha=0.3)
#         plt.plot(distance_mean, F1_mean_pred, color=np.array([c]), label=ifeature)
        if a < -0.8:
            plt.axline((distance_mean[0], F1_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=3)
        else:
            plt.axline((distance_mean[0], F1_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=0.8)
        # plt.axline((distance_mean[0], F1_mean_pred[0]), (distance_mean[-1], F1_mean_pred[-1]), color=np.array([c]), label=ifeature)
    plt.legend(ncol=2)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_per_feature_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_per_feature_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def plot_distance_F1_drop_per_class_per_component(testsets, distance_dfs, train_F1_txt_path, F1_txt_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting correlation of distance and F1 drop per component per class (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    
    df = dataframe_distance_f1_per_feature(testsets, distance_dfs, train_F1_txt_path, F1_txt_paths, threshold)
    
    if class_filter == 'yes':
        df_ID = df[df['testset'] == 'ID_test']
        dropped_classes_1 = df_ID[df_ID['F1'] < 0.75]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        for iclass in dropped_classes:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)

    df_slopes = pd.DataFrame(index=np.unique(df['class'].values), columns=df.columns.values[6:-1])
    random.seed(100)
    colors = distinctipy.get_colors(len(df.columns.values[6:-1]), pastel_factor=0.7)
    for iclass in np.unique(df['class'].values):
        testsets_class = df[df['class'] == iclass].testset.values
        if len(testsets_class) < 2:
            continue
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1)
        plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}^{f}$', fontsize=20)
        plt.ylabel('F1 drop ratio', fontsize=20)
        
        for ifeature,c in zip(df.columns.values[6:-1], colors):
            F1_drop_mean = []
            distance_mean = []
            for i, itestset in enumerate(testsets_class):
                F1_drop_mean_testset = np.average(df[(df['testset'] == itestset) & (df['class'] == iclass)]['F1_drop'])
                F1_drop_mean.append(F1_drop_mean_testset)
                
                distance_mean_testset = np.average(df[(df['testset'] == itestset) & (df['class'] == iclass)][ifeature])
                distance_mean.append(distance_mean_testset)
            params, covariance = curve_fit(linear_func, distance_mean, F1_drop_mean)
            a = params[0]
            b = params[1]
#             a, b = np.polyfit(distance_mean, F1_drop_mean, deg=1)
            F1_drop_mean_pred = a * np.array(distance_mean) + b
            # if a >= -5 and a <= 5:
            df_slopes.loc[iclass, ifeature] = a
            
            plt.scatter(distance_mean, F1_drop_mean, c=np.array([c]), alpha=0.3)
            if a > 0.7:
                plt.axline((distance_mean[0], F1_drop_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=3)
            else:
                plt.axline((distance_mean[0], F1_drop_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=0.8)
        
        plt.title(iclass)
        plt.legend(ncol=2)
        plt.tight_layout()

        if PCA == 'yes':
            Path(outpath + 'per_PC_per_class/').mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath + 'per_PC_per_class/' + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_drop_per_feature_threshold_' + str(threshold) + '_' + iclass + '.png', dpi=300)
        elif PCA == 'no':
            Path(outpath + 'per_feature_per_class/').mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath + 'per_feature_per_class/' + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_drop_per_feature_threshold_' + str(threshold) + '_' + iclass + '.png', dpi=300)
        plt.close()

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 1, 1)
    plt.bar(x=range(len(df_slopes.columns)), height=df_slopes.mean(), color='royalblue')
    plt.xticks(range(len(df_slopes.columns)), labels=df_slopes.columns, rotation=45, rotation_mode='anchor', ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Average slope')
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_drop_feature_mean_slope_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_drop_feature_mean_slope_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()
    
def plot_distance_F1_per_class_per_component(testsets, distance_dfs, train_F1_txt_path, F1_txt_paths, outpath, model, feature_or_pixel, PCA, threshold, distance_type, class_filter):
    print('------------plotting correlation of distance and F1 per component per class (PCA: {}, threshold: {}, distance: {}, class_filter: {})------------'.format(PCA, threshold, distance_type, class_filter))
    
    df = dataframe_distance_f1_per_feature(testsets, distance_dfs, train_F1_txt_path, F1_txt_paths, threshold)
    
    if class_filter == 'yes':
        df_ID = df[df['testset'] == 'ID_test']
        dropped_classes_1 = df_ID[df_ID['F1'] < 0.75]['class'].to_list()
        n_testset = df['class'].value_counts()
        dropped_classes_2 = n_testset[n_testset < 4].index.to_list()
        dropped_classes = list(set(dropped_classes_1 + dropped_classes_2))
        for iclass in dropped_classes:
            df = df.drop(df[df['class'] == iclass].index)
            df = df.reset_index(drop=True)

    df_slopes = pd.DataFrame(index=np.unique(df['class'].values), columns=df.columns.values[6:-1])
    random.seed(100)
    colors = distinctipy.get_colors(len(df.columns.values[6:-1]), pastel_factor=0.7)
    for iclass in np.unique(df['class'].values):
        testsets_class = df[df['class'] == iclass].testset.values
        if len(testsets_class) < 2:
            continue
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1)
        plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}^{f}$')
        plt.ylabel('F1-score')
        
        for ifeature,c in zip(df.columns.values[6:-1], colors):
            F1_mean = []
            distance_mean = []
            for i, itestset in enumerate(testsets_class):
                F1_mean_testset = np.average(df[(df['testset'] == itestset) & (df['class'] == iclass)]['F1'])
                F1_mean.append(F1_mean_testset)
                
                distance_mean_testset = np.average(df[(df['testset'] == itestset) & (df['class'] == iclass)][ifeature])
                distance_mean.append(distance_mean_testset)
            params, covariance = curve_fit(linear_func, distance_mean, F1_mean)
            a = params[0]
            b = params[1]
#             a, b = np.polyfit(distance_mean, F1_mean, deg=1)
            F1_mean_pred = a * np.array(distance_mean) + b
            # if a >= -5 and a <= 5:
            df_slopes.loc[iclass, ifeature] = a
            
            plt.scatter(distance_mean, F1_mean, c=np.array([c]), alpha=0.3)
            if a < -0.8:
                plt.axline((distance_mean[0], F1_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=3)
            else:
                plt.axline((distance_mean[0], F1_mean_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=0.8)
        
        plt.title(iclass)
        plt.legend(ncol=2)
        plt.tight_layout()

        if PCA == 'yes':
            Path(outpath + 'per_PC_per_class/').mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath + 'per_PC_per_class/' + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_per_feature_threshold_' + str(threshold) + '_' + iclass + '.png', dpi=300)
        elif PCA == 'no':
            Path(outpath + 'per_feature_per_class/').mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath + 'per_feature_per_class/' + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_per_feature_threshold_' + str(threshold) + '_' + iclass + '.png', dpi=300)
        plt.close()

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 1, 1)
    plt.bar(x=range(len(df_slopes.columns)), height=df_slopes.mean(), color='royalblue')
    plt.xticks(range(len(df_slopes.columns)), labels=df_slopes.columns, rotation=45, rotation_mode='anchor', ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Average slope')
    plt.tight_layout()
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_feature_mean_slope_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_feature_mean_slope_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()

def linear_func(x, a, b):
    return a * x + b

def plot_distance_F1_testset_x(testsets, distance_txt_paths, train_F1_txt_path, F1_txt_paths, outpath, model, feature_or_pixel, PCA, distance_type):    
    print('------------plotting global correlation of distance and F1 (PCA: {}, distance: {})------------'.format(PCA, distance_type))

    df = pd.DataFrame(columns=['distance', 'F1', 'F1_drop', 'Acc', 'Acc_drop'], index=testsets)
    for i, itestset in enumerate(testsets):
        df_distance = pd.read_csv(distance_txt_paths[i], header=None)
        distance = float(df_distance.iloc[0, 0][18:])
        df_F1 = pd.read_csv(F1_txt_paths[i], header=None)
        F1 = float(df_F1.iloc[3, 0])
        acc = float(df_F1.iloc[1, 0])
        df_F1_train = pd.read_csv(train_F1_txt_path, header=None)
        F1_train = float(df_F1_train.iloc[3, 0])
        acc_train = float(df_F1_train.iloc[1, 0])
        F1_drop = 1 - np.divide(F1, F1_train)
        acc_drop = 1 - np.divide(acc, acc_train)
        df.loc[itestset] = [distance, F1, F1_drop, acc, acc_drop]

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$', fontsize=15)
    plt.ylabel('F1 drop ratio', fontsize=15)
    random.seed(100)
    colors = distinctipy.get_colors(len(testsets), pastel_factor=0.7)
    for x, y, c, l in zip(df.distance, df.F1_drop, colors, testsets):
        plt.scatter(x, y, c=np.array([c]), label=l, s=70)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_drop_testset_err_x.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_drop_testset_err_x.png', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$', fontsize=15)
    plt.ylabel('F1-score', fontsize=15)
    for x, y, c, l in zip(df.distance, df.F1, colors, testsets):
        plt.scatter(x, y, c=np.array([c]), label=l, s=70)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_testset_err_x.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_testset_err_x.png', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$', fontsize=15)
    plt.ylabel('Accuracy drop ratio', fontsize=15)
    random.seed(100)
    colors = distinctipy.get_colors(len(testsets), pastel_factor=0.7)
    for x, y, c, l in zip(df.distance, df.Acc_drop, colors, testsets):
        plt.scatter(x, y, c=np.array([c]), label=l, s=70)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_acc_drop_testset_err_x.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_acc_drop_testset_err_x.png', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    for x, y, c, l in zip(df.distance, df.Acc, colors, testsets):
        plt.scatter(x, y, c=np.array([c]), label=l, s=70)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_acc_testset_err_x.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_acc_testset_err_x.png', dpi=300)
    plt.close()


def plot_distance_F1_testset_y(testsets, distance_txt_paths, train_F1_txt_path, F1_txt_paths, outpath, model):    
    print('------------plotting global correlation of abundance distance and F1------------')

    df = pd.DataFrame(columns=['distance', 'F1', 'F1_drop', 'Acc', 'Acc_drop'], index=testsets)
    for i, itestset in enumerate(testsets):
        df_distance = pd.read_csv(distance_txt_paths[i], header=None)
        distance = float(df_distance.iloc[0, 0][18:])
        df_F1 = pd.read_csv(F1_txt_paths[i], header=None)
        F1 = float(df_F1.iloc[3, 0])
        acc = float(df_F1.iloc[1, 0])
        df_F1_train = pd.read_csv(train_F1_txt_path, header=None)
        F1_train = float(df_F1_train.iloc[3, 0])
        acc_train = float(df_F1_train.iloc[1, 0])
        F1_drop = 1 - np.divide(F1, F1_train)
        acc_drop = 1 - np.divide(acc, acc_train)
        df.loc[itestset] = [distance, F1, F1_drop, acc, acc_drop]

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distributional Distance ' + r'$d_y$', fontsize=15)
    plt.ylabel('F1 drop ratio', fontsize=15)
    random.seed(100)
    colors = distinctipy.get_colors(len(testsets), pastel_factor=0.7)
    for x, y, c, l in zip(df.distance, df.F1_drop, colors, testsets):
        plt.scatter(x, y, c=np.array([c]), label=l, s=70)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath + model + '_abundance_distance_F1_drop_testset_err.png', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distributional Distance ' + r'$d_y$', fontsize=15)
    plt.ylabel('F1-score', fontsize=15)
    for x, y, c, l in zip(df.distance, df.F1, colors, testsets):
        plt.scatter(x, y, c=np.array([c]), label=l, s=70)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath + model + '_abundance_distance_F1_testset_err.png', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distributional Distance ' + r'$d_y$', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    for x, y, c, l in zip(df.distance, df.Acc, colors, testsets):
        plt.scatter(x, y, c=np.array([c]), label=l, s=70)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath + model + '_abundance_distance_acc_testset_err.png', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.xlabel('Distributional Distance ' + r'$d_y$', fontsize=15)
    plt.ylabel('Accuracy drop ratio', fontsize=15)
    for x, y, c, l in zip(df.distance, df.Acc_drop, colors, testsets):
        plt.scatter(x, y, c=np.array([c]), label=l, s=70)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=5)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath + model + '_abundance_distance_acc_drop_testset_err.png', dpi=300)
    plt.close()


def plot_distance_F1_x(testsets, distance_dfs, train_F1_txt_path, F1_txt_paths, outpath, model, feature_or_pixel, PCA, distance_type):
    print('------------plotting correlation of distance and F1 per component (PCA: {}, distance: {})------------'.format(PCA, distance_type))
    
    df = pd.DataFrame()
    for i, itestset in enumerate(testsets):
        df_distance = distance_dfs[i]

        df_distance.columns = df_distance.columns.str.replace(r'^m(\d{2})$', r'$M\1$', regex=True)
        df_distance.columns = df_distance.columns.str.replace(r'^mu(\d{2})$', r'$μ\1$', regex=True)
        df_distance.columns = df_distance.columns.str.replace(r'^nu(\d{2})$', r'$η\1$', regex=True)
        df_distance.columns = df_distance.columns.str.replace(r'^hu(\d{2})$', r'$I\1$', regex=True)

        # distance = df_distance.iloc[0].to_list()
        df_F1 = pd.read_csv(F1_txt_paths[i], header=None)
        F1 = float(df_F1.iloc[3, 0])
        df_F1_train = pd.read_csv(train_F1_txt_path, header=None)
        F1_train = float(df_F1_train.iloc[3, 0])
        F1_drop = 1 - np.divide(F1, F1_train)
        df_distance['F1'] = F1
        df_distance['F1_drop'] = F1_drop
        df_distance.rename(index={0 : itestset}, inplace=True)
        df = pd.concat([df, df_distance])
    
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1 drop ratio')
    random.seed(100)
    colors = distinctipy.get_colors(len(df.columns.values[:-2]), pastel_factor=0.7)
    for ifeature, c in zip(df.columns.values[:-2], colors):
        plt.scatter(df[ifeature], df.F1_drop, c=np.array([c]), alpha=0.3)
        params, covariance = curve_fit(linear_func, df[ifeature], df.F1_drop)
        a = params[0]
        b = params[1]
        F1_drop_pred = a * np.array(df[ifeature]) + b
        if a > 0.7:
            plt.axline((df[ifeature][0], F1_drop_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=3)
        else:
            plt.axline((df[ifeature][0], F1_drop_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=0.8)
    plt.legend(ncol=2)
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_drop_per_feature_x.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_drop_per_feature_x.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 1, 1)
    plt.xlabel('Hellinger distance ' + r'$D_\mathcal{H}$')
    plt.ylabel('F1-score')
    for ifeature, c in zip(df.columns.values[:-2], colors):
        plt.scatter(df[ifeature], df.F1, c=np.array([c]), alpha=0.3)
        params, covariance = curve_fit(linear_func, df[ifeature], df.F1)
        a = params[0]
        b = params[1]
        F1_pred = a * np.array(df[ifeature]) + b
        if a < -0.8:
            plt.axline((df[ifeature][0], F1_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=3)
        else:
            plt.axline((df[ifeature][0], F1_pred[0]), slope=a, color=np.array([c]), label=ifeature, linewidth=0.8)
    plt.legend(ncol=2)
    if PCA == 'yes':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_PCA_' + distance_type + '_distance_F1_per_feature_x.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + model + '_' + feature_or_pixel + '_' + distance_type + '_distance_F1_per_feature_x.png', dpi=300)
    plt.close()

def plot_distance(testsets, distance_txt_paths, outpath, feature_or_pixel, PCA, threshold, distance_type):
    df = pd.DataFrame()
    for i in range(len(testsets)):
        df_distance = preprocess_distance_txt(distance_txt_paths[i])    
        df_distance_testset = pd.DataFrame(columns=['class','testset', 'distance'], index=range(len(df_distance)))
        df_distance_testset['class'] = df_distance['class']
        df_distance_testset['testset'] = testsets[i]
        df_distance_testset['distance'] = df_distance['distance']
        df = pd.concat([df, df_distance_testset])
    df = df.reset_index(drop=True)

    classes = np.unique(df['class'])
    df_plot = pd.DataFrame(columns=['ID_test', 'OOD1', 'OOD2', 'OOD3', 'OOD4', 'OOD5'], index=classes)

    for index, row in df.iterrows():
        df_plot.loc[row['class'], row['testset']] = row['distance']
    
    plt.figure(figsize=(10, 8))
    plt.xlabel('class')
    plt.ylabel('Distance to training set ' + '(' + feature_or_pixel + ')')
    plt.grid(alpha=0.5)
    plt.xticks(range(len(classes)), labels=classes, rotation=45, rotation_mode='anchor', ha='right')
    for column in df_plot:
        plt.scatter(x=range(len(df_plot)), y=df_plot[column], label=column, s=10)
    plt.scatter(x=range(len(df_plot)), y=df_plot.mean(axis=1), label='average', s=70, c='red')
    plt.legend(loc=8, bbox_to_anchor=(0.5, -0.3), fancybox=True, ncol=7)
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if PCA == 'yes':
        plt.savefig(outpath + feature_or_pixel + '_PCA_' + distance_type + '_distance_class_threshold_' + str(threshold) + '.png', dpi=300)
    elif PCA == 'no':
        plt.savefig(outpath + feature_or_pixel + '_' + distance_type + '_distance_class_threshold_' + str(threshold) + '.png', dpi=300)
    plt.close()



parser = argparse.ArgumentParser(description='Plot Domain distance versus F1 score')
parser.add_argument('-testsets', nargs='*', help='list of testset names')
parser.add_argument('-distance_class_xlsx_paths', nargs='*', help='list of distance xlsx files')
parser.add_argument('-distance_txt_paths', nargs='*', help='list of distance txt files')
parser.add_argument('-train_F1_txt_path', help='F1 txt file of training set')
parser.add_argument('-F1_txt_paths', nargs='*', help='list of F1 txt files')
parser.add_argument('-outpath', help='path for saving the figure')
parser.add_argument('-model', help='model that being tested')
parser.add_argument('-feature_or_pixel', choices=['feature', 'pixel'], help='using feature distance or pixel distance')
parser.add_argument('-PCA', choices=['yes', 'no'], help='the distance is calculated with PCA or not')
parser.add_argument('-threshold', type=int, default=0, help='threshold of image number to select analysed classes')
parser.add_argument('-distance_type', choices=['Hellinger', 'Wasserstein', 'KL', 'Theta', 'Chi', 'I', 'Imax', 'Imagewise', 'Mahalanobis'], help='type of distribution distance')
parser.add_argument('-class_filter', choices=['yes', 'no'], help='whether to filter out less important classes')
parser.add_argument('-global_x', choices=['yes', 'no'], default='no', help='PCA on data over all classes or not')
parser.add_argument('-abundance', choices=['yes', 'no'], default='no', help='analysis on abundance distance or not')
args = parser.parse_args()

if __name__ == '__main__':
    if args.abundance == 'yes':
        plot_distance_F1_testset_y(args.testsets, args.distance_txt_paths, args.train_F1_txt_path, args.F1_txt_paths, args.outpath, args.model)
    elif args.abundance == 'no':
        if args.global_x == 'no':
            # plot_distance_F1_scatter(args.testsets, args.distance_txt_paths, args.F1_txt_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type)
            # plot_distance(args.testsets, args.distance_txt_paths, args.outpath, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type)
            # plot_distance_F1_err(args.testsets, args.distance_txt_paths, args.F1_txt_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type)
            plot_distance_F1_testset(args.testsets, args.distance_txt_paths, args.F1_txt_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)
            plot_distance_F1_class(args.testsets, args.distance_txt_paths, args.F1_txt_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)
            plot_distance_F1_drop_testset(args.testsets, args.distance_txt_paths, args.train_F1_txt_path, args.F1_txt_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)
            plot_distance_F1_drop_class(args.testsets, args.distance_txt_paths, args.train_F1_txt_path, args.F1_txt_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)

            if args.PCA == 'no':
                distance_dfs = []
                for i in args.distance_class_xlsx_paths:
                    df_distance = pd.read_excel(i, index_col=0)
                    distance_dfs.append(df_distance)
                plot_distance_F1(args.testsets, distance_dfs, args.train_F1_txt_path, args.F1_txt_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)
                plot_distance_F1_drop(args.testsets, distance_dfs, args.train_F1_txt_path, args.F1_txt_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)
                plot_distance_F1_per_class_per_component(args.testsets, distance_dfs, args.train_F1_txt_path, args.F1_txt_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)
                plot_distance_F1_drop_per_class_per_component(args.testsets, distance_dfs, args.train_F1_txt_path, args.F1_txt_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.threshold, args.distance_type, args.class_filter)

        elif args.global_x == 'yes':
            if args.PCA == 'yes':
                plot_distance_F1_testset_x(args.testsets, args.distance_txt_paths, args.train_F1_txt_path, args.F1_txt_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.distance_type)
            elif args.PCA == 'no':
                distance_dfs = []
                for i in args.distance_class_xlsx_paths:
                    df_distance = pd.read_excel(i, index_col=0)
                    distance_dfs.append(df_distance)
                plot_distance_F1_x(args.testsets, distance_dfs, args.train_F1_txt_path, args.F1_txt_paths, args.outpath, args.model, args.feature_or_pixel, args.PCA, args.distance_type)