import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math

from utils_analysis.lib import feature_extraction as fe

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 15})
# plt.rc('axes', labelsize=15)
# plt.rc('legend', fontsize=15)
# plt.rc('figure', titlesize=15) 


def ConcatAllClasses(datapath, nice_feature):

    '''Concatenate features of all images (all classes) in a dataset.'''

    list_class = os.listdir(datapath)
    list.sort(list_class)
    df_all_feat = pd.DataFrame()

    for iclass in list_class:
        class_datapath = datapath + iclass + '/'
        df_class_feat = fe.ConcatAllFeatures(class_datapath, nice_feature)
        df_class_feat['class'] = iclass
        df_all_feat = pd.concat([df_all_feat, df_class_feat], ignore_index=True)

    return df_all_feat


def Standardize(dataframe):

    '''Standardize an input pandas dataframe.'''

    data = dataframe.iloc[:, :-1].values
    data = StandardScaler().fit_transform(data)
    cols = dataframe.columns[:-1]
    # cols = [i + '_standardized' for i in cols]
    df_standardized = pd.DataFrame(data, columns=cols)
    df_standardized['class'] = dataframe['class']

    return df_standardized


def Normalize_minmax(dataframe):

    '''Normalize an input pandas dataframe.'''

    df = dataframe.drop(columns=['class'])
    df_normalized=(df-df.min())/(df.max()-df.min())
    df_normalized['class'] = dataframe['class']

    return df_normalized


def Normalize_std(dataframe):

    '''Normalize an input pandas dataframe.'''

    df = dataframe.drop(columns=['class'])
    df_normalized = (df-df.mean())/df.std()
    df_normalized['class'] = dataframe['class']

    return df_normalized

def Center(dataframe):

    '''Normalize an input pandas dataframe.'''

    df = dataframe.drop(columns=['class'])
    df_centered = df-df.mean()
    df_centered['class'] = dataframe['class']

    return df_centered


def Rescale_log(dataframe):

    df = dataframe.drop(columns=['class'])
    # df_rescale = (math.e - 1) * np.divide((df - df.min()), (df.max() - df.min())) + 1
    df_rescale = (10 - 1) * np.divide((df - df.min()), (df.max() - df.min())) + 1

    # data_rescale_log = np.log(df_rescale)
    data_rescale_log = np.emath.logn(10, df_rescale)
    df_rescale_log = pd.DataFrame(data=data_rescale_log, columns=dataframe.columns[:-1])

    # df_rescale_log_center = df_rescale_log - df_rescale_log.mean()

    df_rescale_log['class'] = dataframe['class']

    return df_rescale_log


def PrincipalComponentAnalysis(dataframe, n_components):

    '''Principal component analysis on a dataframe.'''

    pca = PCA(n_components=n_components)
    pca.fit(dataframe.iloc[:, :-1].values)

    return pca


def PCA_train_val_test(dataframe, pca):

    '''Implement PCA on in-distribution datasets.'''

    principal_components = pca.transform(dataframe.iloc[:, :-1].values)
    df_pca_split = pd.DataFrame(data=principal_components, columns=['principal_component_{}'.format(i+1) for i in range(np.shape(principal_components)[1])])
    df_pca_split['class'] = dataframe['class']

    return df_pca_split


def PCA_OOD(dataframe_OOD, pca):

    '''Implement PCA on out-of-distribution datasets.'''

    principal_components = pca.transform(dataframe_OOD.iloc[:, :-1].values)
    df_pca_OOD = pd.DataFrame(data=principal_components, columns=['principal_component_{}'.format(i+1) for i in range(np.shape(principal_components)[1])])
    df_pca_OOD['class'] = dataframe_OOD['class']

    return df_pca_OOD


parser = argparse.ArgumentParser(description='Principal component analysis on datasets')
parser.add_argument('-Zoolake2_datapath', help='path of the Zoolake2 dataset')
parser.add_argument('-in_distribution_datapaths', nargs='*', help='paths of the in-domain datasets, in an order of: train_val_test')
parser.add_argument('-OOD_datapaths', nargs='*', help='paths of the out-of-distribution datasets')
parser.add_argument('-outpath', help='path for saving output csv')
parser.add_argument('-n_components', type=float, help='number of principal components')
parser.add_argument('-nice_feature', choices=['yes', 'no'], help='only use nice features or not')
parser.add_argument('-global_x', choices=['yes', 'no'], default='no', help='PCA on data over all classes or not')
args = parser.parse_args()


if __name__ == '__main__':

    if args.n_components >= 1:
        args.n_components = int(args.n_components)

    if args.global_x == 'no':
        df = ConcatAllClasses(args.Zoolake2_datapath, args.nice_feature)
        df_train = ConcatAllClasses(args.in_distribution_datapaths[0], args.nice_feature)
        df_val = ConcatAllClasses(args.in_distribution_datapaths[1], args.nice_feature)
        df_test = ConcatAllClasses(args.in_distribution_datapaths[2], args.nice_feature)
        df_pca_train = pd.DataFrame()
        df_pca_val = pd.DataFrame()
        df_pca_test = pd.DataFrame()
        df_OODs = []
        df_pca_OODs = []
        for i in range(len(args.OOD_datapaths)):
            df_OOD = ConcatAllClasses(args.OOD_datapaths[i], args.nice_feature)
            df_pca_OOD = pd.DataFrame()
            df_OODs.append(df_OOD)
            df_pca_OODs.append(df_pca_OOD)
            
        for iclass in np.unique(df['class'].values):
            df_class = df[df['class'] == iclass]
            df_class_standardized = Standardize(df_class)
            pca = PrincipalComponentAnalysis(df_class_standardized, n_components=args.n_components)
            outpath_class = args.outpath + 'PCA_class/' + iclass + '/'
            Path(outpath_class).mkdir(parents=True, exist_ok=True)
            df_components = pd.DataFrame(data=pca.components_, index=['principal_component_{}'.format(i+1) for i in range(len(pca.explained_variance_))], columns=df_class.columns[:-1])
            df_components.to_excel(outpath_class + 'PCA_components_feature.xlsx')
            Path(outpath_class + 'components/').mkdir(parents=True, exist_ok=True)
            for i, ipc in enumerate(df_components.index):
                plt.figure(figsize=(15, 10))
                plt.ylabel('Component loading')
                # plt.bar(x=range(len(df_components.columns)), height=abs(df_components.loc[ipc]))
                plt.bar(x=range(len(df_components.columns)), height=df_components.loc[ipc])
                plt.xticks(range(len(df_components.columns)), df_components.columns, rotation=45, rotation_mode='anchor', ha='right')
                plt.tight_layout()
                plt.savefig(outpath_class + 'components/' + 'PCA_components_feature_PC_' + str(i+1) + '.png')
                plt.close()

            # df_explained_variance = pd.DataFrame(data=pca.explained_variance_, index=['principal_component_{}'.format(i+1) for i in range(len(pca.explained_variance_))])
            # df_explained_variance.to_excel(outpath_class + 'PCA_explained_variance_feature.xlsx')

            df_explained_variance_ratio = pd.DataFrame(data=pca.explained_variance_ratio_, index=['principal_component_{}'.format(i+1) for i in range(len(pca.explained_variance_))])
            df_explained_variance_ratio.to_excel(outpath_class + 'PCA_explained_variance_ratio_feature.xlsx')

            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Number of components')
            plt.ylabel('Explained variance ratio')
            plt.grid()
            plt.tight_layout()
            plt.savefig(outpath_class + 'PCA_explained_variance_ratio_feature.png')
            plt.close()

            df_train_class = df_train[df_train['class'] == iclass].reset_index(drop=True)
            df_val_class = df_val[df_val['class'] == iclass].reset_index(drop=True)
            df_test_class = df_test[df_test['class'] == iclass].reset_index(drop=True)
            df_train_class_standardized = Standardize(df_train_class)
            df_val_class_standardized = Standardize(df_val_class)
            df_test_class_standardized = Standardize(df_test_class)
            df_pca_train_class = PCA_train_val_test(df_train_class_standardized, pca)
            df_pca_val_class = PCA_train_val_test(df_val_class_standardized, pca)
            df_pca_test_class = PCA_train_val_test(df_test_class_standardized, pca)
            df_pca_train_class.to_csv(outpath_class + 'PCA_train_feature.csv')
            df_pca_val_class.to_csv(outpath_class + 'PCA_val_feature.csv')
            df_pca_test_class.to_csv(outpath_class + 'PCA_test_feature.csv')
            df_pca_train = pd.concat([df_pca_train, df_pca_train_class], ignore_index=True)
            df_pca_val = pd.concat([df_pca_val, df_pca_val_class], ignore_index=True)
            df_pca_test = pd.concat([df_pca_test, df_pca_test_class], ignore_index=True)

            for i in range(len(args.OOD_datapaths)):
                df_OOD = df_OODs[i]
                df_OOD_class = df_OOD[df_OOD['class'] == iclass].reset_index(drop=True)
                if len(df_OOD_class) == 0:
                    continue
                df_OOD_class_standardized = Standardize(df_OOD_class)
                df_pca_OOD_class = PCA_OOD(df_OOD_class_standardized, pca)
                df_pca_OOD_class.to_csv(outpath_class + 'PCA_OOD{}_feature.csv'.format(i + 1))
                df_pca_OODs[i] = pd.concat([df_pca_OODs[i], df_pca_OOD_class], ignore_index=True)


        df_pca_train.to_csv(args.outpath + 'PCA_train_feature.csv')
        df_pca_val.to_csv(args.outpath + 'PCA_val_feature.csv')
        df_pca_test.to_csv(args.outpath + 'PCA_test_feature.csv')
        for i in range(len(args.OOD_datapaths)):
            df_pca_OODs[i].to_csv(args.outpath + 'PCA_OOD{}_feature.csv'.format(i + 1))

    elif args.global_x == 'yes':
        df = ConcatAllClasses(args.Zoolake2_datapath, args.nice_feature)
        df_standardized = Standardize(df)
        pca = PrincipalComponentAnalysis(df_standardized, n_components=args.n_components)
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        df_components = pd.DataFrame(data=pca.components_, index=['principal_component_{}'.format(i+1) for i in range(len(pca.explained_variance_))], columns=df.columns[:-1])
        df_components.to_excel(args.outpath + 'PCA_components_feature.xlsx')

        Path(args.outpath + 'components/').mkdir(parents=True, exist_ok=True)
        for i, ipc in enumerate(df_components.index):
            plt.figure(figsize=(15, 10))
            plt.ylabel('Component loading')
            plt.bar(x=range(len(df_components.columns)), height=df_components.loc[ipc])
            plt.xticks(range(len(df_components.columns)), df_components.columns, rotation=45, rotation_mode='anchor', ha='right')
            plt.tight_layout()
            plt.savefig(args.outpath + 'components/' + 'PCA_components_feature_PC_' + str(i+1) + '.png')
            plt.close()

        # df_explained_variance = pd.DataFrame(data=pca.explained_variance_, index=['principal_component_{}'.format(i+1) for i in range(len(pca.explained_variance_))])
        # df_explained_variance.to_excel(args.outpath + 'PCA_explained_variance_feature.xlsx')

        df_explained_variance_ratio = pd.DataFrame(data=pca.explained_variance_ratio_, index=['principal_component_{}'.format(i+1) for i in range(len(pca.explained_variance_))])
        df_explained_variance_ratio.to_excel(args.outpath + 'PCA_explained_variance_ratio_feature.xlsx')

        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance ratio')
        plt.grid()
        plt.tight_layout()
        plt.savefig(args.outpath + 'PCA_explained_variance_ratio_feature.png')

        df_train = ConcatAllClasses(args.in_distribution_datapaths[0], args.nice_feature)
        df_val = ConcatAllClasses(args.in_distribution_datapaths[1], args.nice_feature)
        df_test = ConcatAllClasses(args.in_distribution_datapaths[2], args.nice_feature)
        df_train_standardized = Standardize(df_train)
        df_val_standardized = Standardize(df_val)
        df_test_standardized = Standardize(df_test)
        df_pca_train = PCA_train_val_test(df_train_standardized, pca)
        df_pca_val = PCA_train_val_test(df_val_standardized, pca)
        df_pca_test = PCA_train_val_test(df_test_standardized, pca)
        df_pca_train.to_csv(args.outpath + 'PCA_train_feature.csv')
        df_pca_val.to_csv(args.outpath + 'PCA_val_feature.csv')
        df_pca_test.to_csv(args.outpath + 'PCA_test_feature.csv')

        for i in range(len(args.OOD_datapaths)):
            df_OOD = ConcatAllClasses(args.OOD_datapaths[i], args.nice_feature)
            df_OOD_standardized = Standardize(df_OOD)
            df_pca_OOD = PCA_OOD(df_OOD_standardized, pca)
            df_pca_OOD.to_csv(args.outpath + 'PCA_OOD{}_feature.csv'.format(i + 1))


    # df_standardized = Standardize(df)
    # # df_normalized = Normalize_minmax(df)
    # # df_normalized = Normalize_std(df)
    # # df_normalized = Center(df)
    # # df_normalized = Rescale_log(df)

    # pca, df_pca = PrincipalComponentAnalysis(df_standardized, n_components=args.n_components)
    # # pca, df_pca = PrincipalComponentAnalysis(df_normalized, n_components=args.n_components)
    # # pca, df_pca = PrincipalComponentAnalysis(df, n_components=args.n_components)


    # # loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    # # df_loadings = pd.DataFrame(data=loadings.T, index=['principal_component_{}'.format(i+1) for i in range(args.n_components)], columns=df_normalized.columns[:-1])
    # # df_loadings.to_excel(args.outpath + 'PCA_loadings_feature.xlsx')

    # # pca, df_pca = PrincipalComponentAnalysis(df, n_components=args.n_components)
    # Path(args.outpath).mkdir(parents=True, exist_ok=True)
    # df_pca.to_csv(args.outpath + 'PCA_Zoolake2_feature.csv')

    # # np.savetxt(args.outpath + 'PCA_explained_variance_ratio_feature.txt', pca.explained_variance_ratio_)
    # # np.savetxt(args.outpath + 'PCA_components_feature.txt', pca.components_)

    # df_components = pd.DataFrame(data=pca.components_, index=['principal_component_{}'.format(i+1) for i in range(args.n_components)], columns=df.columns[:-1])
    # df_components.to_excel(args.outpath + 'PCA_components_feature.xlsx')

    # Path(args.outpath + 'components/').mkdir(parents=True, exist_ok=True)
    # for i, ipc in enumerate(df_components.index):
    #     plt.figure(figsize=(15, 10))
    #     plt.ylabel('Component loading')
    #     # plt.bar(x=range(len(df_components.columns)), height=abs(df_components.loc[ipc]))
    #     plt.bar(x=range(len(df_components.columns)), height=df_components.loc[ipc])
    #     plt.xticks(range(len(df_components.columns)), df_components.columns, rotation=45, rotation_mode='anchor', ha='right')
    #     plt.tight_layout()
    #     plt.savefig(args.outpath + 'components/' + 'PCA_components_feature_PC_' + str(i+1) + '.png')
    #     plt.close()

    # df_explained_variance = pd.DataFrame(data=pca.explained_variance_, index=['principal_component_{}'.format(i+1) for i in range(args.n_components)])
    # df_explained_variance.to_excel(args.outpath + 'PCA_explained_variance_feature.xlsx')

    # df_explained_variance_ratio = pd.DataFrame(data=pca.explained_variance_ratio_, index=['principal_component_{}'.format(i+1) for i in range(args.n_components)])
    # df_explained_variance_ratio.to_excel(args.outpath + 'PCA_explained_variance_ratio_feature.xlsx')

    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of components')
    # plt.ylabel('Explained variance ratio')
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(args.outpath + 'PCA_explained_variance_ratio_feature.png')


    # df_train = ConcatAllClasses(args.in_distribution_datapaths[0], args.nice_feature)
    # df_val = ConcatAllClasses(args.in_distribution_datapaths[1], args.nice_feature)
    # df_test = ConcatAllClasses(args.in_distribution_datapaths[2], args.nice_feature)
    # df_train_standardized = Standardize(df_train)
    # df_val_standardized = Standardize(df_val)
    # df_test_standardized = Standardize(df_test)
    # # df_train_normalized = Normalize_minmax(df_train)
    # # df_val_normalized = Normalize_minmax(df_val)
    # # df_test_normalized = Normalize_minmax(df_test)
    # # df_train_normalized = Normalize_std(df_train)
    # # df_val_normalized = Normalize_std(df_val)
    # # df_test_normalized = Normalize_std(df_test)
    # # df_train_normalized = Center(df_train)
    # # df_val_normalized = Center(df_val)
    # # df_test_normalized = Center(df_test)
    # # df_train_normalized = Rescale_log(df_train)
    # # df_val_normalized = Rescale_log(df_val)
    # # df_test_normalized = Rescale_log(df_test)

    # df_pca_train = PCA_train_val_test(df_train_standardized, pca)
    # df_pca_val = PCA_train_val_test(df_val_standardized, pca)
    # df_pca_test = PCA_train_val_test(df_test_standardized, pca)
    # # df_pca_train = PCA_train_val_test(df_train_normalized, pca)
    # # df_pca_val = PCA_train_val_test(df_val_normalized, pca)
    # # df_pca_test = PCA_train_val_test(df_test_normalized, pca)
    # # df_pca_train = PCA_train_val_test(df_train, pca)
    # # df_pca_val = PCA_train_val_test(df_val, pca)
    # # df_pca_test = PCA_train_val_test(df_test, pca)
    # df_pca_train.to_csv(args.outpath + 'PCA_train_feature.csv')
    # df_pca_val.to_csv(args.outpath + 'PCA_val_feature.csv')
    # df_pca_test.to_csv(args.outpath + 'PCA_test_feature.csv')

    # for i in range(len(args.OOD_datapaths)):
    #     df_OOD = ConcatAllClasses(args.OOD_datapaths[i], args.nice_feature)
    #     df_OOD_standardized = Standardize(df_OOD)
    #     # df_OOD_normalized = Normalize_minmax(df_OOD)
    #     # df_OOD_normalized = Normalize_std(df_OOD)
    #     # df_OOD_normalized = Center(df_OOD)
    #     # df_OOD_normalized = Rescale_log(df_OOD)

    #     df_pca_OOD = PCA_OOD(df_OOD_standardized, pca)
    #     # df_pca_OOD = PCA_OOD(df_OOD_normalized, pca)
    #     # df_pca_OOD = PCA_OOD(df_OOD, pca)
    #     df_pca_OOD.to_csv(args.outpath + 'PCA_OOD{}_feature.csv'.format(i + 1))
