import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
from distinctipy import distinctipy
from scipy.stats import sem

import argparse
from pathlib import Path

random.seed(100)
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 15})
# plt.rc('axes', labelsize=15)
# plt.rc('legend', fontsize=15)
# plt.rc('figure', titlesize=15) 


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

def baseline_performance(datasets, F1_txt_paths, outpath):
    F1_macro_mean = []
    F1_macro_sem = []
    # F1_micro_mean = []
    # F1_micro_sem = []
    classes = []
    for i in range(len(datasets)):
        df_performance = preprocess_F1_txt(F1_txt_paths[i])
        macro_F1 = df_performance.F1.mean()
        sem_macro_F1 = sem(df_performance.F1)
        # micro_F1 = np.average(df_performance.F1, weights=df_performance['support'].values)
        # var_micro_F1 = np.divide(np.average((df_performance.F1 - micro_F1) ** 2, weights=df_performance['support'].values), len(df_performance.F1) - 1)
        # sem_micro_F1 = np.sqrt(var_micro_F1)
        # # sem_micro_F1 = np.sqrt(np.divide(var_micro_F1, len(df_performance.F1)))
        F1_macro_mean.append(macro_F1)
        F1_macro_sem.append(sem_macro_F1)
        # F1_micro_mean.append(micro_F1)
        # F1_micro_sem.append(sem_micro_F1)
        classes = classes + df_performance['class'].values.tolist()
    classes = np.unique(classes)
    
    df_F1 = pd.DataFrame(index=classes, columns=datasets)
    for i, itestset in enumerate(datasets):
        df_performance = preprocess_F1_txt(F1_txt_paths[i])
        df_performance = df_performance.set_index('class')
        for iclass in df_performance.index:
            df_F1.loc[iclass, itestset] = df_performance.loc[iclass, 'F1']

    df_recall = pd.DataFrame(index=classes, columns=datasets)
    for i, itestset in enumerate(datasets):
        df_performance = preprocess_F1_txt(F1_txt_paths[i])
        df_performance = df_performance.set_index('class')
        for iclass in df_performance.index:
            df_recall.loc[iclass, itestset] = df_performance.loc[iclass, 'recall']

    ## Macro average F1-score
    plt.figure(figsize=(5, 5))
    plt.xlabel('Dataset')
    plt.ylabel('F1-score')
    plt.grid(axis='y', which='major', alpha=0.5)
    plt.xticks(range(len(datasets)), labels=datasets, rotation=45, rotation_mode='anchor', ha='right')
    # plt.scatter(x=range(len(datasets)), y=F1_mean)
    plt.errorbar(x=range(len(datasets)), y=F1_macro_mean, yerr=F1_macro_sem, fmt='s', capsize=5, color='seagreen', label='Macro_F1')
    # plt.errorbar(x=range(len(datasets)), y=F1_micro_mean, yerr=F1_micro_sem, fmt='s', capsize=5, color='royalblue', label='Micro_F1')
    # plt.legend()
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath + 'baseline_performance.png', dpi=300)
    plt.close()

    ## F1-score per class
    # plt.figure(figsize=(12, 7))
    plt.figure(figsize=(11, 11))
    plt.subplot(1, 1, 1)
    plt.xlabel('Dataset', fontsize=30)
    plt.ylabel('F1-score', fontsize=30)
    plt.ylim((-0.1, 1.1))
    plt.xticks(range(len(datasets)+1), labels=['ID_train', 'ID_test', 'micro-OOD', 'macro-OOD', 'OOD1', 'OOD2', 'OOD3', 'OOD4', 'OOD5', 'OOD6', 'OOD7', 'OOD8', 'OOD9', 'OOD10'], rotation=45, rotation_mode='anchor', ha='right')
    random.seed(100)
    colors = distinctipy.get_colors(len(classes), pastel_factor=0.7)
    markers = ['o', '^', 's', '*', 'p', 'd']
    colors_new = ['b', 'g', 'r', 'c', 'm', 'y']
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        # if iclass == 'nauplius' or iclass == 'eudiaptomus' or iclass == 'unknown':
        #     plt.plot(range(3), df_F1.loc[iclass][:3], linewidth=2, label=iclass, color=c, ls='', marker='*', ms=20)
        #     plt.plot(range(3, 4), np.mean(df_F1.loc[iclass][-10:]), linewidth=2, color=c, ls='', marker='*', ms=20)
        #     plt.plot(range(4, len(datasets)+1), df_F1.loc[iclass][-10:], linewidth=2, label=iclass+ ' (on OOD cells)', color=c, ls='', marker='o', ms=12)
        # # # else:
        # # #     plt.plot(range(len(datasets)), df_F1.loc[iclass, :], label=iclass, color=c, alpha=0.3, ls='', marker='.')
    
        plt.plot(range(3), df_F1.loc[iclass][:3], linewidth=2, label=iclass, color=colors_new[i//6], ls='', marker=markers[i%6], ms=10, alpha=0.5)
        plt.plot(range(3, 4), np.mean(df_F1.loc[iclass][-10:]), linewidth=2, color=colors_new[i//6], ls='', marker=markers[i%6], ms=10, alpha=0.5)
    #     plt.plot(range(4, len(datasets)+1), df_F1.loc[iclass][-10:], linewidth=2, color=c, ls='', marker='o')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    # # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=5)
    # plt.legend()
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath + 'baseline_performance_F1_per_class.png', dpi=300)
    plt.close()

    ## Recall per class
    # plt.figure(figsize=(12, 7))
    plt.figure(figsize=(11, 11))
    plt.subplot(1, 1, 1)
    plt.xlabel('Dataset', fontsize=30)
    plt.ylabel('Recall', fontsize=30)
    plt.ylim((-0.1, 1.1))
    plt.xticks(range(len(datasets)+1), labels=['ID_train', 'ID_test', 'micro-OOD', 'macro-OOD', 'OOD1', 'OOD2', 'OOD3', 'OOD4', 'OOD5', 'OOD6', 'OOD7', 'OOD8', 'OOD9', 'OOD10'], rotation=45, rotation_mode='anchor', ha='right')
    random.seed(100)
    colors = distinctipy.get_colors(len(classes), pastel_factor=0.7)
    for i, (iclass, c) in enumerate(zip(classes, colors)):
        # if iclass == 'nauplius' or iclass == 'eudiaptomus' or iclass == 'unknown':
        #     plt.plot(range(3), df_recall.loc[iclass][:3], linewidth=2, label=iclass, color=c, ls='', marker='*', ms=20)
        #     plt.plot(range(3, 4), np.mean(df_recall.loc[iclass][-10:]), linewidth=2, color=c, ls='', marker='*', ms=20)
        #     plt.plot(range(4, len(datasets)+1), df_recall.loc[iclass][-10:], linewidth=2, label=iclass+ ' (on OOD cells)', color=c, ls='', marker='o', ms=12)
        # # # else:
        # # #     plt.plot(range(len(datasets)), df_recall.loc[iclass, :], label=iclass, color=c, alpha=0.3, ls='', marker='.')
    
        plt.plot(range(3), df_recall.loc[iclass][:3], linewidth=2, label=iclass, color=colors_new[i//6], ls='', marker=markers[i%6], ms=10, alpha=0.5)
        plt.plot(range(3, 4), np.mean(df_recall.loc[iclass][-10:]), linewidth=2, color=colors_new[i//6], ls='', marker=markers[i%6], ms=10, alpha=0.5)
    #     plt.plot(range(4, len(datasets)+1), df_recall.loc[iclass][-10:], linewidth=2, color=c, ls='', marker='o')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    # # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=5)
    # plt.legend()
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath + 'baseline_performance_recall_per_class.png', dpi=300)
    plt.close()

def prediction_confidence(datasets, confidence_csv_paths, outpath):
    conf_mean = []
    conf_sem = []
    for i in range(len(datasets)):
        confs = np.loadtxt(confidence_csv_paths[i])
        mean_confs = confs.mean()
        sem_confs = sem(confs)
        conf_mean.append(mean_confs)
        conf_sem.append(sem_confs)
    
    plt.figure(figsize=(5, 5))
    plt.xlabel('Dataset')
    plt.ylabel('Prediction confidence')
    plt.grid(axis='y', which='major', alpha=0.5)
    plt.xticks(range(len(datasets)), labels=datasets, rotation=45, rotation_mode='anchor', ha='right')
    # plt.scatter(x=range(len(datasets)), y=conf_mean)
    plt.errorbar(x=range(len(datasets)), y=conf_mean, yerr=conf_sem, fmt='s', capsize=5, color='royalblue')
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath + 'prediction_confidence.png', dpi=300)
    plt.close()

def abundance_distance_performance(testsets, train_F1_txt_path, test_F1_txt_paths, outpath):
    df_train_performance = preprocess_F1_txt(train_F1_txt_path)
    abundance_train = df_train_performance['support'].values / df_train_performance['support'].sum()
    df_abundance = pd.DataFrame(index=df_train_performance['class'].values)
    df_abundance['ID_train'] = abundance_train
    abundance_distances = []
    accuracies = []
    macro_F1s = []
    macro_recalls = []
    for i, itestset in enumerate(testsets):
        df_test_performance  = preprocess_F1_txt(test_F1_txt_paths[i])
        abundance_test = df_test_performance['support'].values / df_test_performance['support'].sum()
        df_abundance[itestset] = 0
        for j, iclass in enumerate(df_test_performance['class'].values):
            df_abundance.loc[iclass, itestset] = abundance_test[j]
        distance = abundance_distance(df_abundance['ID_train'], df_abundance[itestset])
        abundance_distances.append(distance)
        accuracies.append(np.average(df_test_performance['recall'], weights=abundance_test))
        macro_F1s.append(df_test_performance['F1'].mean())
        macro_recalls.append(df_test_performance['recall'].mean())
    
    plt.figure(figsize=(5, 5))
    plt.xlabel('Abundance distance')
    plt.ylabel('Performance metric')
    plt.scatter(x=abundance_distances, y=accuracies, label='Accuracy')
    plt.scatter(x=abundance_distances, y=macro_F1s, label='Macro F1')
    # plt.scatter(x=abundance_distances, y=macro_recalls, label='Macro recall')
    plt.legend()
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath + 'abundance_distance_performance.png', dpi=300)
    plt.close()

def abundance_distance(abundance_percentage_1, abundance_percentage_2):
    p1, p2 = abundance_percentage_1, abundance_percentage_2
    p1_normalized = p1 / np.linalg.norm(p1)
    p2_normalized = p2 / np.linalg.norm(p2)
    assert np.absolute(np.linalg.norm(p1_normalized)) - 1 < 10e-8
    assert np.absolute(np.linalg.norm(p2_normalized)) - 1 < 10e-8
    overlap = np.dot(p1_normalized, p2_normalized)
    abundance_distance = 1 - overlap

    return abundance_distance

def preprocess_bias_xlsx(bias_xlsx_path):
    df_bias = pd.read_excel(bias_xlsx_path)
    df_bias = df_bias.rename(columns={df_bias.columns[0]: 'class'})
    # df_bias['APE'] = abs(np.divide(df_bias['Bias'], df_bias['Ground_truth']))
    # df_bias['SAPE'] = abs(np.divide(df_bias['Bias'], df_bias['Ground_truth'] + df_bias['Predict']))
    # df_bias['NAE'] = abs(np.divide(df_bias['Bias'], np.sum(df_bias['Ground_truth'])))

    return df_bias

def population_scatter(testsets, test_bias_xlsx_paths, outpath):
    classes = []
    for i in range(len(testsets)):
        df_bias = preprocess_bias_xlsx(test_bias_xlsx_paths[i])
        classes = classes + df_bias['class'].values.tolist()
    classes = np.unique(classes)

    df_true = pd.DataFrame(index=classes, columns=testsets)
    df_pred = pd.DataFrame(index=classes, columns=testsets)
    # df_pred_AC = pd.DataFrame(index=classes, columns=testsets)
    # df_pred_PCC = pd.DataFrame(index=classes, columns=testsets)
    # df_pred_PAC = pd.DataFrame(index=classes, columns=testsets)

    for i, itestset in enumerate(testsets):
        df_bias = preprocess_bias_xlsx(test_bias_xlsx_paths[i])
        df_bias = df_bias.set_index('class')
        for iclass in df_bias.index:
            df_true.loc[iclass, itestset] = df_bias.loc[iclass, 'Ground_truth']
            df_pred.loc[iclass, itestset] = df_bias.loc[iclass, 'Predict']
            # df_pred_AC.loc[iclass, itestset] = df_bias.loc[iclass, 'AC']
            # df_pred_PCC.loc[iclass, itestset] = df_bias.loc[iclass, 'PCC']
            # df_pred_PAC.loc[iclass, itestset] = df_bias.loc[iclass, 'PAC']

    fig, ax = plt.subplots(figsize=(8, 8))
    # plt.figure(figsize=(9, 10))
    ax.set_xlabel(r'$N_\mathrm{true}$', fontsize=21)
    ax.set_ylabel(r'$N_\mathrm{pred}$', fontsize=21)
    random.seed(100)
    # colors = distinctipy.get_colors(len(classes), pastel_factor=0.7)
    colors = distinctipy.get_colors(len(testsets), pastel_factor=0.7)

    ax2 = fig.add_axes([.21, .65, .3, .3])
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_xlim(0, 1000)
    # ax.set_ylim(0, 1000)
    ax2.set_xlabel(r'$N_\mathrm{true}$')
    ax2.set_ylabel(r'$N_\mathrm{pred}$')
    ax2.axline((0, 0), slope=1, c='red', ls='--', lw=1)
    ax.axline((0, 0), (1000, 1000), c='red', ls='--', lw=1)

    for itestset, c in zip(testsets, colors):
        ax.scatter(df_true[itestset], df_pred[itestset], label=itestset, color=c)
        # ax.scatter(df_true, df_pred_AC, label='AC', color='orange')
        # ax.scatter(df_true, df_pred_PCC, label='PCC', color='darkviolet')
        # ax.scatter(df_true, df_pred_PAC, label='PAC', color='olive')
        ax2.scatter(df_true[itestset], df_pred[itestset], label=itestset, color=c, s=7)
        # ax2.scatter(df_true, df_pred_AC, label='AC', color='orange')
        # ax2.scatter(df_true, df_pred_PCC, label='PCC', color='darkviolet')
        # ax2.scatter(df_true, df_pred_PAC, label='PAC', color='olive')

    # for iclass, c in zip(classes, colors):
    #     ax.scatter(df_true.loc[iclass], df_pred.loc[iclass], label=iclass, color=c)
    #     ax.scatter(df_true.loc[iclass], df_pred_AC.loc[iclass], label=iclass, color=c, marker='+')
    #     ax2.scatter(df_true.loc[iclass], df_pred.loc[iclass], label=iclass, color=c)
    #     ax2.scatter(df_true.loc[iclass], df_pred_AC.loc[iclass], label=iclass, color=c, marker='+')

    ax.legend(loc=4)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=10)
    ax2.set_xlim([-50, 2000])
    ax2.set_ylim([-50, 2000])
    ax.set_xlim([1, 2000])
    ax.set_ylim([1, 2000])
    # plt.axis('square')
    # ax.set_aspect('equal', 'datalim')
    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath + 'population_scatter.png', dpi=300)
    plt.close()

parser = argparse.ArgumentParser(description='Plot model performances')
parser.add_argument('-datasets', nargs='*', help='list of dataset names')
parser.add_argument('-testsets', nargs='*', help='list of testset names')
parser.add_argument('-F1_txt_paths', nargs='*', help='list of F1 txt files')
parser.add_argument('-train_F1_txt_path', help='F1 txt files on training dataset')
parser.add_argument('-test_F1_txt_paths', nargs='*', help='list of F1 txt files on testset')
parser.add_argument('-test_bias_xlsx_paths', nargs='*', help='list of bias xlsx files on testset')
parser.add_argument('-confidence_csv_paths', nargs='*', help='list of prediction confidence csv files')
parser.add_argument('-outpath', help='path for saving the figure')
args = parser.parse_args()

if __name__ == '__main__':
    baseline_performance(args.datasets, args.F1_txt_paths, args.outpath)
    # prediction_confidence(args.datasets, args.confidence_csv_paths, args.outpath)
    # abundance_distance_performance(args.testsets, args.train_F1_txt_path, args.test_F1_txt_paths, args.outpath)
    # population_scatter(args.testsets, args.test_bias_xlsx_paths, args.outpath)