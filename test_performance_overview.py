import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem
import random
from distinctipy import distinctipy


plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 15})
# plt.rc('axes', labelsize=15)
# plt.rc('legend', fontsize=15)
# plt.rc('figure', titlesize=15) 


def plot_performance_overview(model_name, test_dataset, accuracy, f1_score, BC, NMAE, outpath, remove_0):
    plt.figure(figsize=(30, 14))
    plt.subplot(1, 4, 1)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.yticks(np.arange(len(model_name)), labels=model_name)
    plt.title('Accuracy')
    for i in range(len(model_name)):
        for j in range(len(test_dataset)):
            text = plt.text(j, i, format(accuracy[i, j], '.3f'), ha='center', va='center', color='black')
    plt.imshow(accuracy, cmap='RdYlGn', vmin=0.3, vmax=1.0)
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.yticks(np.arange(len(model_name)), labels=model_name)
    plt.title('F1-score')
    for i in range(len(model_name)):
        for j in range(len(test_dataset)):
            text = plt.text(j, i, format(f1_score[i, j], '.3f'), ha='center', va='center', color='black')
    plt.imshow(f1_score, cmap='RdYlGn', vmin=0.3, vmax=1.0)
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.yticks(np.arange(len(model_name)), labels=model_name)
    plt.title('Bray-Curtis Dissimilarity')
    for i in range(len(model_name)):
        for j in range(len(test_dataset)):
            text = plt.text(j, i, format(BC[i, j], '.3f'), ha='center', va='center', color='black')
    plt.imshow(BC, cmap='RdYlGn_r', vmin=0, vmax=0.3)
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.yticks(np.arange(len(model_name)), labels=model_name)
    plt.title('Normalized Mean Absolute error')
    for i in range(len(model_name)):
        for j in range(len(test_dataset)):
            text = plt.text(j, i, format(NMAE[i, j], '.3f'), ha='center', va='center', color='black')
    plt.imshow(NMAE, cmap='RdYlGn_r')
    plt.colorbar()

    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if remove_0 == 'no':
        plt.savefig(outpath + 'test_performance_overview.png', dpi=300)
    elif remove_0 == 'yes':
        plt.savefig(outpath + 'test_performance_overview_rm_0.png', dpi=300)
    plt.close()


def read_test_report(test_report_file):
    test_report = pd.read_csv(test_report_file)
    accuracy_value = format(float(test_report.iloc[0].item()), '.3f')
    f1_value = format(float(test_report.iloc[2].item()), '.3f')

    return accuracy_value, f1_value

def read_extra_test_report(test_extra_report_file):
    extra_test_report = pd.read_csv(test_extra_report_file)
    bias_value = format(float(extra_test_report.iloc[0].item()), '.3f')
    BC_value = format(float(extra_test_report.iloc[2].item()), '.5f')
    MAE_value = format(float(extra_test_report.iloc[4].item()), '.5f')
    MSE_value = format(float(extra_test_report.iloc[6].item()), '.5f')
    RMSE_value = format(float(extra_test_report.iloc[8].item()), '.5f')
    R2_value = format(float(extra_test_report.iloc[10].item()), '.5f')
    NMAE_value = format(float(extra_test_report.iloc[12].item()), '.5f')
    AE_rm_junk_value = format(float(extra_test_report.iloc[14].item()), '.5f')
    NAE_rm_junk_value = format(float(extra_test_report.iloc[16].item()), '.5f')
    
    return bias_value, BC_value, MAE_value, MSE_value, RMSE_value, R2_value, NMAE_value, AE_rm_junk_value, NAE_rm_junk_value

def performance_matrix(model_performance_paths, testsets, finetuned, ensemble, remove_0, TTA):
    n_model = len(model_performance_paths)
    # n_dataset = len(os.listdir(model_performance_paths[0]))
    n_dataset = len(testsets)
    # test_dataset = os.listdir(model_performance_paths[0])

    accuracy = np.zeros([n_model, n_dataset])
    f1_score = np.zeros([n_model, n_dataset])
    bias = np.zeros([n_model, n_dataset])
    BC = np.zeros([n_model, n_dataset])
    MAE = np.zeros([n_model, n_dataset])
    MSE = np.zeros([n_model, n_dataset])
    RMSE = np.zeros([n_model, n_dataset])
    R2 = np.zeros([n_model, n_dataset])
    NMAE = np.zeros([n_model, n_dataset])
    AE_rm_junk = np.zeros([n_model, n_dataset])
    NAE_rm_junk = np.zeros([n_model, n_dataset])

    for i, imodel_path in enumerate(model_performance_paths):
        # dataset_names = os.listdir(imodel_path)
        dataset_names = testsets
        # dataset_names.sort()
        for j, idataset in enumerate(dataset_names):
            # if TTA == 'no':
            if TTA[i] == 'no':
                test_report_path = imodel_path + '/' + idataset + '/'
            # elif TTA == 'yes':
            elif TTA[i] == 'yes':
                test_report_path = imodel_path + '/' + idataset + '/TTA_result/' 

            if remove_0 == 'no':
                if finetuned == 0:
                    if ensemble[i] == 0:
                        test_report_file = test_report_path + 'Single_test_report_original.txt'
                        extra_test_report_file = test_report_path + 'Single_test_report_extra_original.txt'
                    elif ensemble[i] == 1:
                        test_report_file = test_report_path + 'Ensemble_test_report_geo_mean_original.txt'
                        extra_test_report_file = test_report_path + 'Ensemble_test_report_extra_geo_mean_original.txt'
                if finetuned == 1:
                    if ensemble[i] == 0:
                        test_report_file = test_report_path + 'Single_test_report_tuned.txt'
                        extra_test_report_file = test_report_path + 'Single_test_report_extra_tuned.txt'
                    elif ensemble[i] == 1:
                        test_report_file = test_report_path + 'Ensemble_test_report_geo_mean_tuned.txt'
                        extra_test_report_file = test_report_path + 'Ensemble_test_report_extra_geo_mean_tuned.txt'
                if finetuned == 2:
                    if ensemble[i] == 0:
                        test_report_file = test_report_path + 'Single_test_report_finetuned.txt'
                        extra_test_report_file = test_report_path + 'Single_test_report_extra_finetuned.txt'
                    elif ensemble[i] == 1:
                        test_report_file = test_report_path + 'Ensemble_test_report_geo_mean_finetuned.txt'
                        extra_test_report_file = test_report_path + 'Ensemble_test_report_extra_geo_mean_finetuned.txt'
            elif remove_0 == 'yes':
                if finetuned == 0:
                    if ensemble[i] == 0:
                        test_report_file = test_report_path + 'Single_test_report_rm_0_original.txt'
                        extra_test_report_file = test_report_path + 'Single_test_report_extra_rm_0_original.txt'
                    elif ensemble[i] == 1:
                        test_report_file = test_report_path + 'Ensemble_test_report_rm_0_geo_mean_original.txt'
                        extra_test_report_file = test_report_path + 'Ensemble_test_report_extra_rm_0_geo_mean_original.txt'
                if finetuned == 1:
                    if ensemble[i] == 0:
                        test_report_file = test_report_path + 'Single_test_report_rm_0_tuned.txt'
                        extra_test_report_file = test_report_path + 'Single_test_report_extra_rm_0_tuned.txt'
                    elif ensemble[i] == 1:
                        test_report_file = test_report_path + 'Ensemble_test_report_rm_0_geo_mean_tuned.txt'
                        extra_test_report_file = test_report_path + 'Ensemble_test_report_extra_rm_0_geo_mean_tuned.txt'
                if finetuned == 2:
                    if ensemble[i] == 0:
                        test_report_file = test_report_path + 'Single_test_report_rm_0_finetuned.txt'
                        extra_test_report_file = test_report_path + 'Single_test_report_extra_rm_0_finetuned.txt'
                    elif ensemble[i] == 1:
                        test_report_file = test_report_path + 'Ensemble_test_report_rm_0_geo_mean_finetuned.txt'
                        extra_test_report_file = test_report_path + 'Ensemble_test_report_extra_rm_0_geo_mean_finetuned.txt'

            accuracy_value, f1_value = read_test_report(test_report_file)
            bias_value, BC_value, MAE_value, MSE_value, RMSE_value, R2_value, NMAE_value, AE_rm_junk_value, NAE_rm_junk_value = read_extra_test_report(extra_test_report_file)
            accuracy[i, j], f1_score[i, j] = accuracy_value, f1_value
            bias[i, j], BC[i, j], MAE[i, j], MSE[i, j], RMSE[i, j], R2[i, j], NMAE[i, j], AE_rm_junk[i, j], NAE_rm_junk[i, j] = bias_value, BC_value, MAE_value, MSE_value, RMSE_value, R2_value, NMAE_value, AE_rm_junk_value, NAE_rm_junk_value
    
    return accuracy, f1_score, bias, BC, MAE, MSE, RMSE, R2, NMAE, AE_rm_junk, NAE_rm_junk


def plot_performance_curve(model_name, test_dataset, accuracy, f1_score, BC, NMAE, outpath, remove_0):
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 3, 1)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.title('Accuracy')
    for i, j in zip(accuracy, model_name):
        plt.plot(i, label=j)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.title('F1-score')
    for i, j in zip(f1_score, model_name):
        plt.plot(i, label=j)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.title('Bray-Curtis Dissimilarity')
    for i, j in zip(BC, model_name):
        plt.plot(i, label=j)
    plt.legend()

    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if remove_0 == 'no':
        plt.savefig(outpath + 'test_performance_curves.png', dpi=300)
    elif remove_0 == 'yes':
        plt.savefig(outpath + 'test_performance_curves_rm_0.png', dpi=300)
    plt.close()


def plot_performance_baseline(aug_types, test_dataset, n_OOD_cells, accuracy, f1_score, BC, NMAE, outpath, remove_0):
    n_aug = len(aug_types)
    n_model_aug = len(accuracy)/n_aug
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 4, 1)
    # plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.xticks(np.arange(len(test_dataset)+1), labels=['ID_train', 'ID_test', 'micro-OOD', 'macro-OOD', 'OOD1', 'OOD2', 'OOD3', 'OOD4', 'OOD5', 'OOD6', 'OOD7', 'OOD8', 'OOD9', 'OOD10'], rotation=45, rotation_mode='anchor', ha='right')
    # plt.ylim(0.3, 1)
    plt.title('Accuracy')
    for i, aug_type in enumerate(aug_types):
        accuracy_aug_type = accuracy[int(i*n_model_aug):int((i+1)*n_model_aug)]
        mean_accuracy = np.mean(accuracy_aug_type, axis=0)
        sem_accuracy = sem(accuracy_aug_type, axis=0)
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells-1), y=mean_accuracy[:len(test_dataset)-n_OOD_cells-1], yerr=sem_accuracy[:len(test_dataset)-n_OOD_cells-1], marker='s', capsize=3, ms=10, linestyle='', c='blue', label='ID')
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells-1, len(test_dataset)-n_OOD_cells), y=mean_accuracy[len(test_dataset)-n_OOD_cells-1:len(test_dataset)-n_OOD_cells], yerr=sem_accuracy[len(test_dataset)-n_OOD_cells-1:len(test_dataset)-n_OOD_cells], marker='s', capsize=3, ms=10, linestyle='', c='red', label='micro-OOD')
        y_macro_OOD, yerr_macro_OOD = np.mean(mean_accuracy[len(test_dataset)-n_OOD_cells:]), sem(mean_accuracy[len(test_dataset)-n_OOD_cells:])
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells, len(test_dataset)-n_OOD_cells+1), y=y_macro_OOD, yerr=yerr_macro_OOD, marker='s', capsize=3, ms=10, linestyle='', c='orange', label='macro-OOD')
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells+1, len(test_dataset)+1), y=mean_accuracy[len(test_dataset)-n_OOD_cells:], yerr=sem_accuracy[len(test_dataset)-n_OOD_cells:], marker='s', markerfacecolor='none', capsize=3, ms=10, linestyle='', c='orange', label='OOD cells')
    # plt.legend()

    plt.subplot(1, 4, 2)
    # plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.xticks(np.arange(len(test_dataset)+1), labels=['ID_train', 'ID_test', 'micro-OOD', 'macro-OOD', 'OOD1', 'OOD2', 'OOD3', 'OOD4', 'OOD5', 'OOD6', 'OOD7', 'OOD8', 'OOD9', 'OOD10'], rotation=45, rotation_mode='anchor', ha='right')
    # plt.ylim(0.3, 1)
    plt.title('F1-score')
    for i, aug_type in enumerate(aug_types):
        f1_aug_type = f1_score[int(i*n_model_aug):int((i+1)*n_model_aug)]
        mean_f1 = np.mean(f1_aug_type, axis=0)
        sem_f1 = sem(f1_aug_type, axis=0)
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells-1), y=mean_f1[:len(test_dataset)-n_OOD_cells-1], yerr=sem_f1[:len(test_dataset)-n_OOD_cells-1], marker='s', capsize=3, ms=10, linestyle='', c='blue', label='ID')
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells-1, len(test_dataset)-n_OOD_cells), y=mean_f1[len(test_dataset)-n_OOD_cells-1:len(test_dataset)-n_OOD_cells], yerr=sem_f1[len(test_dataset)-n_OOD_cells-1:len(test_dataset)-n_OOD_cells], marker='s', capsize=3, ms=10, linestyle='', c='red', label='micro-OOD')
        y_macro_OOD, yerr_macro_OOD = np.mean(mean_f1[len(test_dataset)-n_OOD_cells:]), sem(mean_f1[len(test_dataset)-n_OOD_cells:])
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells, len(test_dataset)-n_OOD_cells+1), y=y_macro_OOD, yerr=yerr_macro_OOD, marker='s', capsize=3, ms=10, linestyle='', c='orange', label='macro-OOD')
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells+1, len(test_dataset)+1), y=mean_f1[len(test_dataset)-n_OOD_cells:], yerr=sem_f1[len(test_dataset)-n_OOD_cells:], marker='s', markerfacecolor='none', capsize=3, ms=10, linestyle='', c='orange', label='OOD cells')
    # plt.legend()

    plt.subplot(1, 4, 3)
    # plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.xticks(np.arange(len(test_dataset)+1), labels=['ID_train', 'ID_test', 'micro-OOD', 'macro-OOD', 'OOD1', 'OOD2', 'OOD3', 'OOD4', 'OOD5', 'OOD6', 'OOD7', 'OOD8', 'OOD9', 'OOD10'], rotation=45, rotation_mode='anchor', ha='right')
    plt.title('Bray-Curtis Dissimilarity')
    for i, aug_type in enumerate(aug_types):
        BC_aug_type = BC[int(i*n_model_aug):int((i+1)*n_model_aug)]
        mean_BC = np.mean(BC_aug_type, axis=0)
        sem_BC = sem(BC_aug_type, axis=0)
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells-1), y=mean_BC[:len(test_dataset)-n_OOD_cells-1], yerr=sem_BC[:len(test_dataset)-n_OOD_cells-1], marker='s', capsize=3, ms=10, linestyle='', c='blue', label='ID')
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells-1, len(test_dataset)-n_OOD_cells), y=mean_BC[len(test_dataset)-n_OOD_cells-1:len(test_dataset)-n_OOD_cells], yerr=sem_BC[len(test_dataset)-n_OOD_cells-1:len(test_dataset)-n_OOD_cells], marker='s', capsize=3, ms=10, linestyle='', c='red', label='micro-OOD')
        y_macro_OOD, yerr_macro_OOD = np.mean(mean_BC[len(test_dataset)-n_OOD_cells:]), sem(mean_BC[len(test_dataset)-n_OOD_cells:])
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells, len(test_dataset)-n_OOD_cells+1), y=y_macro_OOD, yerr=yerr_macro_OOD, marker='s', capsize=3, ms=10, linestyle='', c='orange', label='macro-OOD')
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells+1, len(test_dataset)+1), y=mean_BC[len(test_dataset)-n_OOD_cells:], yerr=sem_BC[len(test_dataset)-n_OOD_cells:], marker='s', markerfacecolor='none', capsize=3, ms=10, linestyle='', c='orange', label='OOD cells')
    # plt.legend()

    plt.subplot(1, 4, 4)
    # plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.xticks(np.arange(len(test_dataset)+1), labels=['ID_train', 'ID_test', 'micro-OOD', 'macro-OOD', 'OOD1', 'OOD2', 'OOD3', 'OOD4', 'OOD5', 'OOD6', 'OOD7', 'OOD8', 'OOD9', 'OOD10'], rotation=45, rotation_mode='anchor', ha='right')
    plt.title('Normalized Mean Absolute Error')
    for i, aug_type in enumerate(aug_types):
        NMAE_aug_type = NMAE[int(i*n_model_aug):int((i+1)*n_model_aug)]
        mean_NMAE = np.mean(NMAE_aug_type, axis=0)
        sem_NMAE = sem(NMAE_aug_type, axis=0)
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells-1), y=mean_NMAE[:len(test_dataset)-n_OOD_cells-1], yerr=sem_NMAE[:len(test_dataset)-n_OOD_cells-1], marker='s', capsize=3, ms=10, linestyle='', c='blue', label='ID')
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells-1, len(test_dataset)-n_OOD_cells), y=mean_NMAE[len(test_dataset)-n_OOD_cells-1:len(test_dataset)-n_OOD_cells], yerr=sem_NMAE[len(test_dataset)-n_OOD_cells-1:len(test_dataset)-n_OOD_cells], marker='s', capsize=3, ms=10, linestyle='', c='red', label='micro-OOD')
        y_macro_OOD, yerr_macro_OOD = np.mean(mean_NMAE[len(test_dataset)-n_OOD_cells:]), sem(mean_NMAE[len(test_dataset)-n_OOD_cells:])
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells, len(test_dataset)-n_OOD_cells+1), y=y_macro_OOD, yerr=yerr_macro_OOD, marker='s', capsize=3, ms=10, linestyle='', c='orange', label='macro-OOD')
        plt.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells+1, len(test_dataset)+1), y=mean_NMAE[len(test_dataset)-n_OOD_cells:], yerr=sem_NMAE[len(test_dataset)-n_OOD_cells:], marker='s', markerfacecolor='none', capsize=3, ms=10, linestyle='', c='orange', label='OOD cells')
    # plt.legend()

    plt.legend(loc='upper center', bbox_to_anchor=(-1.3, 1.12), fancybox=True, ncol=4)
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3)
    # plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if remove_0 == 'no':
        plt.savefig(outpath + 'test_performance.png', dpi=300)
    elif remove_0 == 'yes':
        plt.savefig(outpath + 'test_performance_rm_0.png', dpi=300)
    plt.close()


def plot_performance(aug_types, test_dataset, n_OOD_cells, accuracy, f1_score, BC, NMAE, outpath, remove_0):
    n_aug = len(aug_types)
    n_model_aug = len(accuracy)/n_aug

    random.seed(100)
    colors = distinctipy.get_colors(n_aug, pastel_factor=0.7)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
    plt.subplots_adjust(left=0.05, bottom=0.12, right=0.98, top=0.90, wspace=0.15)

    ax1.set_xticks(np.arange(len(test_dataset)+1), labels=['ID_train', 'ID_test', 'micro-OOD', 'macro-OOD', 'OOD1', 'OOD2', 'OOD3', 'OOD4', 'OOD5', 'OOD6', 'OOD7', 'OOD8', 'OOD9', 'OOD10'], rotation=45, rotation_mode='anchor', ha='right')
    # ax1.set_ylim(0.3, 1)
    ax1.set_title('Accuracy')
    for i, (aug_type, c) in enumerate(zip(aug_types, colors)):
        accuracy_aug_type = accuracy[int(i*n_model_aug):int((i+1)*n_model_aug)]
        mean_accuracy = np.mean(accuracy_aug_type, axis=0)
        sem_accuracy = sem(accuracy_aug_type, axis=0)
        ax1.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells), y=mean_accuracy[:len(test_dataset)-n_OOD_cells], yerr=sem_accuracy[:len(test_dataset)-n_OOD_cells], marker='s', capsize=3, ms=10, linestyle='', color=c, label=aug_type)
        y_macro_OOD, yerr_macro_OOD = np.mean(mean_accuracy[len(test_dataset)-n_OOD_cells:]), sem(mean_accuracy[len(test_dataset)-n_OOD_cells:])
        ax1.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells, len(test_dataset)-n_OOD_cells+1), y=y_macro_OOD, yerr=yerr_macro_OOD, marker='s', capsize=3, ms=10, linestyle='', color=c, label=aug_type)
        ax1.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells+1, len(test_dataset)+1), y=mean_accuracy[len(test_dataset)-n_OOD_cells:], yerr=sem_accuracy[len(test_dataset)-n_OOD_cells:], marker='s', markerfacecolor='none', capsize=3, ms=10, linestyle='', color=c, label=aug_type + ' on OOD cells')
    # plt.legend()

    ax2.set_xticks(np.arange(len(test_dataset)+1), labels=['ID_train', 'ID_test', 'micro-OOD', 'macro-OOD', 'OOD1', 'OOD2', 'OOD3', 'OOD4', 'OOD5', 'OOD6', 'OOD7', 'OOD8', 'OOD9', 'OOD10'], rotation=45, rotation_mode='anchor', ha='right')
    # ax2.set_ylim(0.3, 1)
    ax2.set_title('F1-score')
    for i, (aug_type, c) in enumerate(zip(aug_types, colors)):
        f1_aug_type = f1_score[int(i*n_model_aug):int((i+1)*n_model_aug)]
        mean_f1 = np.mean(f1_aug_type, axis=0)
        sem_f1 = sem(f1_aug_type, axis=0)
        ax2.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells), y=mean_f1[:len(test_dataset)-n_OOD_cells], yerr=sem_f1[:len(test_dataset)-n_OOD_cells], marker='s', capsize=3, ms=10, linestyle='', color=c, label=aug_type)
        y_macro_OOD, yerr_macro_OOD = np.mean(mean_f1[len(test_dataset)-n_OOD_cells:]), sem(mean_f1[len(test_dataset)-n_OOD_cells:])
        ax2.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells, len(test_dataset)-n_OOD_cells+1), y=y_macro_OOD, yerr=yerr_macro_OOD, marker='s', capsize=3, ms=10, linestyle='', color=c, label=aug_type)
        ax2.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells+1, len(test_dataset)+1), y=mean_f1[len(test_dataset)-n_OOD_cells:], yerr=sem_f1[len(test_dataset)-n_OOD_cells:], marker='s', markerfacecolor='none', capsize=3, ms=10, linestyle='', color=c, label=aug_type + ' on OOD cells')
    # plt.legend()

    ax3.set_xticks(np.arange(len(test_dataset)+1), labels=['ID_train', 'ID_test', 'micro-OOD', 'macro-OOD', 'OOD1', 'OOD2', 'OOD3', 'OOD4', 'OOD5', 'OOD6', 'OOD7', 'OOD8', 'OOD9', 'OOD10'], rotation=45, rotation_mode='anchor', ha='right')
    ax3.set_title('Bray-Curtis Dissimilarity')
    for i, (aug_type, c) in enumerate(zip(aug_types, colors)):
        BC_aug_type = BC[int(i*n_model_aug):int((i+1)*n_model_aug)]
        mean_BC = np.mean(BC_aug_type, axis=0)
        sem_BC = sem(BC_aug_type, axis=0)
        ax3.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells), y=mean_BC[:len(test_dataset)-n_OOD_cells], yerr=sem_BC[:len(test_dataset)-n_OOD_cells], marker='s', capsize=3, ms=10, linestyle='', color=c, label=aug_type)
        y_macro_OOD, yerr_macro_OOD = np.mean(mean_BC[len(test_dataset)-n_OOD_cells:]), sem(mean_BC[len(test_dataset)-n_OOD_cells:])
        ax3.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells, len(test_dataset)-n_OOD_cells+1), y=y_macro_OOD, yerr=yerr_macro_OOD, marker='s', capsize=3, ms=10, linestyle='', color=c, label=aug_type)
        ax3.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells+1, len(test_dataset)+1), y=mean_BC[len(test_dataset)-n_OOD_cells:], yerr=sem_BC[len(test_dataset)-n_OOD_cells:], marker='s', markerfacecolor='none', capsize=3, ms=10, linestyle='', color=c, label=aug_type + ' on OOD cells')
    # plt.legend()

    ax4.set_xticks(np.arange(len(test_dataset)+1), labels=['ID_train', 'ID_test', 'micro-OOD', 'macro-OOD', 'OOD1', 'OOD2', 'OOD3', 'OOD4', 'OOD5', 'OOD6', 'OOD7', 'OOD8', 'OOD9', 'OOD10'], rotation=45, rotation_mode='anchor', ha='right')
    ax4.set_title('Normalized Mean Absolute Error')
    for i, (aug_type, c) in enumerate(zip(aug_types, colors)):
        NMAE_aug_type = NMAE[int(i*n_model_aug):int((i+1)*n_model_aug)]
        mean_NMAE = np.mean(NMAE_aug_type, axis=0)
        sem_NMAE = sem(NMAE_aug_type, axis=0)
        ax4.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells), y=mean_NMAE[:len(test_dataset)-n_OOD_cells], yerr=sem_NMAE[:len(test_dataset)-n_OOD_cells], marker='s', capsize=3, ms=10, linestyle='', color=c, label=aug_type)
        y_macro_OOD, yerr_macro_OOD = np.mean(mean_NMAE[len(test_dataset)-n_OOD_cells:]), sem(mean_NMAE[len(test_dataset)-n_OOD_cells:])
        ax4.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells, len(test_dataset)-n_OOD_cells+1), y=y_macro_OOD, yerr=yerr_macro_OOD, marker='s', capsize=3, ms=10, linestyle='', color=c)
        ax4.errorbar(x=np.arange(len(test_dataset)-n_OOD_cells+1, len(test_dataset)+1), y=mean_NMAE[len(test_dataset)-n_OOD_cells:], yerr=sem_NMAE[len(test_dataset)-n_OOD_cells:], marker='s', markerfacecolor='none', capsize=3, ms=10, linestyle='', color=c, label=aug_type + ' on OOD cells')
    # plt.legend()

    ax4.legend(loc='upper center', bbox_to_anchor=(-1.3, 1.12), fancybox=True, ncol=8)
    # fig.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if remove_0 == 'no':
        fig.savefig(outpath + 'test_performance.png', dpi=300)
    elif remove_0 == 'yes':
        fig.savefig(outpath + 'test_performance_rm_0.png', dpi=300)
    plt.close(fig)

def plot_performance_AE(aug_types, test_dataset, accuracy, f1_score, BC, outpath, remove_0):
    n_aug = len(aug_types)
    n_model_aug = len(accuracy)/n_aug
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 3, 1)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.title('Normalized Absolute Error')
    for i, aug_type in enumerate(aug_types):
        accuracy_aug_type = accuracy[int(i*n_model_aug):int((i+1)*n_model_aug)]
        mean_accuracy = np.mean(accuracy_aug_type, axis=0)
        sem_accuracy = sem(accuracy_aug_type, axis=0)
        plt.errorbar(x=np.arange(len(test_dataset)), y=mean_accuracy, yerr=sem_accuracy, marker='s', markerfacecolor='none', capsize=3, ms=5, label=aug_type)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.title('Absolute Error w/o junk')
    for i, aug_type in enumerate(aug_types):
        f1_aug_type = f1_score[int(i*n_model_aug):int((i+1)*n_model_aug)]
        mean_f1 = np.mean(f1_aug_type, axis=0)
        sem_f1 = sem(f1_aug_type, axis=0)
        plt.errorbar(x=np.arange(len(test_dataset)), y=mean_f1, yerr=sem_f1, marker='s', markerfacecolor='none', capsize=3, ms=5, label=aug_type)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.xticks(np.arange(len(test_dataset)), labels=test_dataset, rotation=45, rotation_mode='anchor', ha='right')
    plt.title('Normalized Absolute Error w/o junk')
    for i, aug_type in enumerate(aug_types):
        BC_aug_type = BC[int(i*n_model_aug):int((i+1)*n_model_aug)]
        mean_BC = np.mean(BC_aug_type, axis=0)
        sem_BC = sem(BC_aug_type, axis=0)
        plt.errorbar(x=np.arange(len(test_dataset)), y=mean_BC, yerr=sem_BC, marker='s', markerfacecolor='none', capsize=3, ms=5, label=aug_type)
    plt.legend()

    plt.tight_layout()
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if remove_0 == 'no':
        plt.savefig(outpath + 'test_performance_AE.png', dpi=300)
    elif remove_0 == 'yes':
        plt.savefig(outpath + 'test_performance_rm_0_AE.png', dpi=300)
    plt.close()

parser = argparse.ArgumentParser(description='Generate an overview of test performance')
parser.add_argument('-model_names', nargs='*', help='list of model names')
parser.add_argument('-aug_types', nargs='*', help='list of augmentation types')
parser.add_argument('-testsets', nargs='*', help='list of test datasets')
parser.add_argument('-model_performance_paths', nargs='*', help='list of performance paths for each model')
parser.add_argument('-outpath', help='path for saving the overview')
parser.add_argument('-finetuned', type=int, help='0 for original, 1 for tuned, 2 for finetuned')
# parser.add_argument('-ensemble', type=int, help='0 for single model, 1 for ensembled model')
parser.add_argument('-remove_0', choices=['yes', 'no'], default='no', help='remove 0 support classes or not')
# parser.add_argument('-TTA', choices=['yes', 'no'], default='no', help='using TTA results or not')
parser.add_argument('-ensemble', nargs="+", type=int, help='0 for single model, 1 for ensembled model')
parser.add_argument('-TTA', nargs='*', help='using TTA results or not')
args = parser.parse_args()

if __name__ == '__main__':
    n_OOD_cells = 10
    accuracy, f1_score, bias, BC, MAE, MSE, RMSE, R2, NMAE, AE_rm_junk, NAE_rm_junk = performance_matrix(args.model_performance_paths, args.testsets, args.finetuned, args.ensemble, args.remove_0, args.TTA)
    model_name = args.model_names
    outpath = args.outpath
    aug_types = args.aug_types
    # plot_performance_overview(model_name, args.testsets, accuracy, f1_score, BC, NMAE, outpath, args.remove_0)
    # plot_performance_curve(model_name, test_dataset, accuracy, f1_score, BC, NMAE, outpath, args.remove_0)
    # plot_performance_baseline(aug_types, args.testsets, n_OOD_cells, accuracy, f1_score, BC, NMAE, outpath, args.remove_0)
    plot_performance(aug_types, args.testsets, n_OOD_cells, accuracy, f1_score, BC, NMAE, outpath, args.remove_0)

    # plot_performance_AE(aug_types, test_dataset, NMAE, AE_rm_junk, NAE_rm_junk, outpath, args.remove_0)
