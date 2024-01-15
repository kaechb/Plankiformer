###########
# IMPORTS #
###########

import argparse
import os
import pathlib
import pickle
import sys
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score, recall_score, roc_curve, confusion_matrix
from scipy.stats import gmean

plt.rcParams['figure.dpi'] = 300
# plt.rcParams.update({'font.size': 15})
# plt.rc('axes', labelsize=15)
# plt.rc('legend', fontsize=15)
# plt.rc('figure', titlesize=15) 

class LoadEnsembleParameters:
    def __init__(self, initMode='default', verbose=True):
        self.fsummary = None
        self.tt = None
        self.verbose = verbose
        self.params = None
        self.paramsDict = None
        self.data = None
        self.trainSize = None
        self.testSize = None
        self.model = None
        self.opt = None
        self.SetParameters(mode=initMode)

        return

    def SetParameters(self, mode='default'):
        """ default, from args"""
        if mode == 'default':
            self.ReadArgs(string=None)
        elif mode == 'args':
            self.ReadArgs(string=sys.argv[1:])
        else:
            print('Unknown parameter mode', mode)
            raise NotImplementedError
        return

    def ReadArgs(self, string=None):
        if string is None:
            string = ""

        parser = argparse.ArgumentParser(description='Create Dataset')

        parser.add_argument('-main_model_dir',
                            default='/local/kyathasr/Plankiformer/out/phyto_super_class/',
                            help="Main directory where the model is stored")
        parser.add_argument('-main_model',
                            default='/local/kyathasr/Plankiformer/out/phyto_super_class/',
                            help="Main directory where the model is stored")
        parser.add_argument('-outpath', default='./out/Ensemble/', help="directory where you want the output saved")

        parser.add_argument('-finetune', type=int, default=0, help='Choose "0" or "1" or "2" for finetuning')
        parser.add_argument('-model_dirs', nargs='*',
                            default=['./data/'],
                            help="Directories with the model.")
        parser.add_argument('-ens_type', type=int, default=1,
                            help="choose between either arithmetic mean (=1) or geometric mean (=2)")
        parser.add_argument('-threshold', type=float, default=0.0, 
                            help='threshold of confidence for abstention')

        args = parser.parse_args(string)

        args.outpath = args.outpath + '/'

        self.params = args

        if self.verbose:
            print(args)

        return

    def CreateOutDir(self):
        """ Create a unique output directory, and put inside it a file with the simulation parameters """
        pathlib.Path(self.params.outpath).mkdir(parents=True, exist_ok=True)
        self.WriteParams()
        return

    def WriteParams(self):
        """ Writes a txt file with the simulation parameters """
        self.fsummary = open(self.params.outpath + '/params.txt', 'w')
        print(self.params, file=self.fsummary)
        self.fsummary.flush()

        ''' Writes the same simulation parameters in binary '''
        np.save(self.params.outpath + '/params.npy', self.params)
        return

    def UpdateParams(self, **kwargs):
        """ Updates the parameters given in kwargs, and updates params.txt"""
        self.paramsDict = vars(self.params)
        if kwargs is not None:
            for key, value in kwargs.items():
                self.paramsDict[key] = value
        self.CreateOutDir()
        self.WriteParams()

        return

    def get_ensemble_performance(self):
        print('Main model directory: {}'.format(self.params.main_model_dir))
        classes_dir1 = self.params.main_model_dir + '/classes.npy'
        classes_dir2 = self.params.main_model_dir + '/classes.pt'

        if os.path.exists(classes_dir1):
            classes = np.load(self.params.main_model_dir + '/classes.npy')
        elif os.path.exists(classes_dir2):
            classes = torch.load(classes_dir2)
        else:
            classes = ('airplane', 'automobile', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        DEIT_GTLabel_sorted = []
        DEIT_GTLabel_indices = []
        DEIT_PredLabel_sorted = []
        DEIT_Prob_sorted = []

        for model_path in self.params.model_dirs:
            DEIT_model = []
            if self.params.finetune == 0:
                DEIT_model = pd.read_pickle(model_path + '/Single_GT_Pred_GTLabel_PredLabel_PredLabelCorrected_Prob_original.pickle')
                name2 = 'original'
            elif self.params.finetune == 1:
                DEIT_model = pd.read_pickle(model_path + '/Single_GT_Pred_GTLabel_PredLabel_PredLabelCorrected_Prob_tuned.pickle')
                name2 = 'tuned'
            elif self.params.finetune == 2:
                DEIT_model = pd.read_pickle(model_path + '/Single_GT_Pred_GTLabel_PredLabel_PredLabelCorrected_Prob_finetuned.pickle')
                name2 = 'finetuned'
            else:
                print(' Please Select correct finetuning parameters')

            DEIT_01_GTLabel = DEIT_model[2]
            DEIT_01_PredLabel = DEIT_model[3]
            DEIT_01_Prob = DEIT_model[4]

            # DEIT_01_GTLabel_sorted = np.sort(DEIT_01_GTLabel)
            # DEIT_01_GTLabel_indices = np.argsort(DEIT_01_GTLabel)
            # DEIT_01_PredLabel_sorted = DEIT_01_PredLabel[DEIT_01_GTLabel_indices]
            # DEIT_01_Prob_sorted = DEIT_01_Prob[DEIT_01_GTLabel_indices]

            DEIT_GTLabel_sorted.append(DEIT_01_GTLabel)
            # DEIT_GTLabel_indices.append(DEIT_01_GTLabel_indices)
            DEIT_PredLabel_sorted.append(DEIT_01_PredLabel)
            DEIT_Prob_sorted.append(DEIT_01_Prob)

        Ens_DEIT_label = []
        Ens_DEIT = []
        name = ''
        if self.params.ens_type == 1:
            Ens_DEIT = sum(DEIT_Prob_sorted) / len(DEIT_Prob_sorted)
            Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
            Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))],
                                      dtype=object)
            name = 'arth'
        elif self.params.ens_type == 2:
            Ens_DEIT = gmean(DEIT_Prob_sorted)
            Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
            Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))],
                                      dtype=object)
            name = 'geo'
        else:
            print("Choose correct ensemble type")


        # Ens_DEIT_corrected_label = copy.deepcopy(Ens_DEIT_label)

        # first_indices = Ens_DEIT.argsort()[:, -1]
        # Ens_confs = [Ens_DEIT[i][first_indices[i]] for i in range(len(first_indices))]

        # for i in range(len(Ens_confs)):
        #     if Ens_confs[i] < self.params.threshold:
        #         Ens_DEIT_corrected_label[i] = 'unknown'

        # if self.params.threshold > 0:
        #     print('I am using threshold value as : {}'.format(self.params.threshold))

        #     accuracy_model = accuracy_score(DEIT_GTLabel_sorted[0], Ens_DEIT_corrected_label)
        #     clf_report = classification_report(DEIT_GTLabel_sorted[0], Ens_DEIT_corrected_label)
        #     clf_report_rm_0 = classification_report(DEIT_GTLabel_sorted[0], Ens_DEIT_corrected_label, labels=np.unique(DEIT_GTLabel_sorted[0]))
        #     f1 = f1_score(DEIT_GTLabel_sorted[0], Ens_DEIT_corrected_label, average='macro')
        #     f1_rm_0 = f1_score(DEIT_GTLabel_sorted[0], Ens_DEIT_corrected_label, average='macro', labels=np.unique(DEIT_GTLabel_sorted[0]))

        #     f = open(self.params.outpath + 'Ensemble_test_report_' + name2 + name + '_thresholded.txt', 'w')
        #     f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
        #                                                                                           clf_report))
        #     f.close()

        #     ff = open(self.params.outpath + 'Ensemble_test_report_rm_0_' + name2 + name + '_thresholded.txt', 'w')
        #     ff.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1_rm_0,
        #                                                                                           clf_report_rm_0))
        #     ff.close()
            

        Ens_DEIT_corrected_label = copy.deepcopy(Ens_DEIT_label)
        GT_label = copy.deepcopy(DEIT_GTLabel_sorted[0])

        first_indices = Ens_DEIT.argsort()[:, -1]
        Ens_confs = [Ens_DEIT[i][first_indices[i]] for i in range(len(first_indices))]

        for i in range(len(Ens_confs)):
            if Ens_confs[i] < self.params.threshold:
                Ens_DEIT_corrected_label[i] = 'abstention'
        DEIT_corrected_label = [x for x in Ens_DEIT_corrected_label if x != 'abstention']
        GT_label = [GT_label[i] for i, x in enumerate(Ens_DEIT_corrected_label) if x != 'abstention']

        if self.params.threshold > 0:
            print('I am using threshold value as : {}'.format(self.params.threshold))

            accuracy_model = accuracy_score(GT_label, DEIT_corrected_label)
            clf_report = classification_report(GT_label, DEIT_corrected_label)
            clf_report_rm_0 = classification_report(GT_label, DEIT_corrected_label, labels=np.unique(GT_label))
            f1 = f1_score(GT_label, DEIT_corrected_label, average='macro')
            f1_rm_0 = f1_score(GT_label, DEIT_corrected_label, average='macro', labels=np.unique(GT_label))

            f = open(self.params.outpath + 'Ensemble_test_report_' + name2 + name + '_thresholded.txt', 'w')
            f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                                  clf_report))
            f.close()

            ff = open(self.params.outpath + 'Ensemble_test_report_rm_0_' + name2 + name + '_thresholded.txt', 'w')
            ff.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1_rm_0,
                                                                                                  clf_report_rm_0))
            ff.close()

        print('Accuracy:  {}'.format(round(accuracy_score(DEIT_GTLabel_sorted[0], Ens_DEIT_label), 3)))
        print('F1-score:  {}'.format(round(f1_score(DEIT_GTLabel_sorted[0], Ens_DEIT_label, average='macro'), 3)))
        print(classification_report(DEIT_GTLabel_sorted[0], Ens_DEIT_label, digits=2))

        accuracy_model = accuracy_score(DEIT_GTLabel_sorted[0], Ens_DEIT_label)
        clf_report = classification_report(DEIT_GTLabel_sorted[0], Ens_DEIT_label)
        clf_report_rm_0 = classification_report(DEIT_GTLabel_sorted[0], Ens_DEIT_label, labels=np.unique(DEIT_GTLabel_sorted[0]))
        f1 = f1_score(DEIT_GTLabel_sorted[0], Ens_DEIT_label, average='macro')
        f1_rm_0 = f1_score(DEIT_GTLabel_sorted[0], Ens_DEIT_label, average='macro', labels=np.unique(DEIT_GTLabel_sorted[0]))

        Pred_PredLabel_Prob = [DEIT_GTLabel_sorted[0], Ens_DEIT_label, Ens_DEIT]
        with open(self.params.outpath + '/Ensemble_models_GTLabel_PredLabel_Prob_' + name + '.pickle', 'wb') as cw:
            pickle.dump(Pred_PredLabel_Prob, cw)

        f = open(self.params.outpath + 'Ensemble_test_report_' + name + '_mean_' + name2 + '.txt', 'w')
        f.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1,
                                                                                            clf_report))
        f.close()

        ff = open(self.params.outpath + 'Ensemble_test_report_rm_0_' + name + '_mean_' + name2 + '.txt', 'w')
        ff.write('\n Accuracy\n\n{}\n\nF1 Score\n\n{}\n\nClassification Report\n\n{}\n'.format(accuracy_model, f1_rm_0,
                                                                                            clf_report_rm_0))
        ff.close()

        ID_result = pd.read_pickle(self.params.main_model + '/GT_Pred_GTLabel_PredLabel_prob_model_' + name2 + '.pickle')
        bias, BC, MAE, MSE, RMSE, R2, NMAE, AE_rm_junk, NAE_rm_junk, df_count = extra_metrics(DEIT_GTLabel_sorted[0], Ens_DEIT_label, Ens_DEIT, ID_result)
        fff = open(self.params.outpath + 'Ensemble_test_report_extra_' + name + '_mean_' + name2 + '.txt', 'w')
        fff.write('\nbias\n\n{}\n\nBC\n\n{}\n\nMAE\n\n{}\n\nMSE\n\n{}\n\nRMSE\n\n{}\n\nR2\n\n{}\n\nNMAE\n\n{}\n\nAE_rm_junk\n\n{}\n\nNAE_rm_junk\n\n{}\n'.format(bias, BC, MAE, MSE, RMSE, R2, NMAE, AE_rm_junk, NAE_rm_junk))
        fff.close()

        df_count.to_excel(self.params.outpath + 'Population_count.xlsx', index=True, header=True)

        plt.figure(figsize=(8, 6))
        plt.subplot(1, 1, 1)
        plt.xlabel('Class', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        width = 0.5
        x = np.arange(0, len(df_count) * 2, 2)
        x1 = x - width / 2
        x2 = x + width / 2
        plt.bar(x1, df_count['Ground_truth'], width=0.5, label='Ground_truth')
        plt.bar(x2, df_count['Predict'], width=0.5, label='Prediction')
        plt.xticks(x, df_count.index, rotation=45, rotation_mode='anchor', ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.params.outpath + 'Population_count.png', dpi=300)
        plt.close()


def extra_metrics(GT_label, Pred_label, Pred_prob, ID_result):

    list_class = list(set(np.unique(GT_label)).union(set(np.unique(Pred_label))))
    list_class.sort()
    df_count_Pred_GT = pd.DataFrame(index=list_class, columns=['Predict', 'Ground_truth'])

    GT_label_ID = ID_result[2].tolist()
    Pred_label_ID = ID_result[3].tolist()
    Pred_prob_ID = ID_result[4]

    list_class_ID = np.unique(GT_label_ID).tolist()
    list_class_ID.sort()
    df_prob = pd.DataFrame(index=list_class_ID, columns=['prob'])
    for i in range(len(list_class_ID)):
        df_prob.iloc[i] = np.sum(Pred_prob[:, i])

    df_prob_ID_all = pd.DataFrame(data=Pred_prob_ID, columns=list_class_ID)

    CC = []
    AC = []
    PCC = []
    PAC = []

    Pred_label = Pred_label.tolist()
    GT_label = GT_label.tolist()
    for iclass in list_class:
        df_count_Pred_GT.loc[iclass, 'Predict'] = Pred_label.count(iclass)
        df_count_Pred_GT.loc[iclass, 'Ground_truth'] = GT_label.count(iclass)

        class_CC = Pred_label.count(iclass)
        CC.append(class_CC)

        true_copy, pred_copy = GT_label_ID.copy(), Pred_label_ID.copy()
        for i in range(len(GT_label_ID)):
            if GT_label_ID[i] == iclass:
                true_copy[i] = 1
            else:
                true_copy[i] = 0
            if Pred_label_ID[i] == iclass:
                pred_copy[i] = 1
            else:
                pred_copy[i] = 0
        tn, fp, fn, tp = confusion_matrix(true_copy, pred_copy).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        class_AC = (class_CC - (fpr * len(Pred_label))) / (tpr - fpr)
        AC.append(class_AC)

        class_PCC = df_prob.loc[iclass, 'prob']
        PCC.append(class_PCC)

        df_prob_ID = pd.DataFrame()
        df_prob_ID['Pred_label'] = Pred_label_ID
        df_prob_ID['GT_label'] = GT_label_ID
        df_prob_ID['Pred_prob'] = df_prob_ID_all[iclass]
        tpr_prob = np.sum(df_prob_ID[(df_prob_ID['GT_label'] == iclass) & (df_prob_ID['Pred_label'] == iclass)]['Pred_prob']) / (tp + fn)
        fpr_prob = np.sum(df_prob_ID[(df_prob_ID['GT_label'] != iclass) & (df_prob_ID['Pred_label'] == iclass)]['Pred_prob']) / (tn + fp)
        class_PAC = (class_PCC - (fpr_prob * len(Pred_label))) / (tpr_prob - fpr_prob)
        PAC.append(class_PAC)

    df_percentage_Pred_GT = df_count_Pred_GT.div(df_count_Pred_GT.sum(axis=0), axis=1)
    df_count_Pred_GT['Bias'] = df_count_Pred_GT['Predict'] - df_count_Pred_GT['Ground_truth']
    df_count_Pred_GT['CC'], df_count_Pred_GT['AC'], df_count_Pred_GT['PCC'], df_count_Pred_GT['PAC'] = CC, AC, PCC, PAC

    df_count_Pred_GT_rm_junk = df_count_Pred_GT.drop(['dirt', 'unknown', 'unknown_plankton'], errors='ignore')
    df_count_Pred_GT_rm_junk = df_count_Pred_GT_rm_junk.drop(df_count_Pred_GT_rm_junk[df_count_Pred_GT_rm_junk['Ground_truth'] == 0].index)

    df_count_Pred_GT_rm_0 = df_count_Pred_GT.drop(df_count_Pred_GT[df_count_Pred_GT['Ground_truth'] == 0].index)

    bias = np.sum(df_count_Pred_GT['Predict'] - df_count_Pred_GT['Ground_truth']) / df_count_Pred_GT.shape[0]
    BC = np.sum(np.abs(df_count_Pred_GT['Predict'] - df_count_Pred_GT['Ground_truth'])) / np.sum(np.abs(df_count_Pred_GT['Predict'] + df_count_Pred_GT['Ground_truth']))
    MAE = mean_absolute_error(df_count_Pred_GT['Ground_truth'], df_count_Pred_GT['Predict'])
    MSE = mean_squared_error(df_count_Pred_GT['Ground_truth'], df_count_Pred_GT['Predict'])
    RMSE = np.sqrt(MSE)
    R2 = r2_score(df_count_Pred_GT['Ground_truth'], df_count_Pred_GT['Predict'])

    AE_rm_junk = np.sum(np.abs(df_count_Pred_GT_rm_junk['Predict'] - df_count_Pred_GT_rm_junk['Ground_truth']))
    NAE_rm_junk = np.sum(np.divide(np.abs(df_count_Pred_GT_rm_junk['Predict'] - df_count_Pred_GT_rm_junk['Ground_truth']), df_count_Pred_GT_rm_junk['Ground_truth']))
    NMAE = np.mean(np.divide(np.abs(df_count_Pred_GT_rm_0['Predict'] - df_count_Pred_GT_rm_0['Ground_truth']), df_count_Pred_GT_rm_0['Ground_truth']))

    return bias, BC, MAE, MSE, RMSE, R2, NMAE, AE_rm_junk, NAE_rm_junk, df_count_Pred_GT

if __name__ == '__main__':
    print('\n Running Ensemble', sys.argv[0], sys.argv[1:])

    # Loading Input parameters
    train_params = LoadEnsembleParameters(initMode='args')
    train_params.CreateOutDir()
    train_params.get_ensemble_performance()
