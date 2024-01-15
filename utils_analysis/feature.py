from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils_analysis.lib.distance_metrics as dm

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 15})
# plt.rc('axes', labelsize=15)
# plt.rc('legend', fontsize=15)
# plt.rc('figure', titlesize=15) 


def PlotFeatureDistribution(feature_files, outpath, selected_features, n_bins_feature, adaptive_bins, data_labels, image_threshold, distance_type):

    print('-----------------Now plotting feature distribution for each class and each selected feature.-----------------')

    df_1 = pd.read_csv(feature_files[0], index_col=0)
    df_2 = pd.read_csv(feature_files[1], index_col=0)

    list_class_1 = np.unique(df_1['class'])
    list_class_2 = np.unique(df_2['class'])
    list_class_rep = list(set(list_class_1) & set(list_class_2))
    list.sort(list_class_rep)

    if selected_features == ['all']:
        selected_features = df_1.columns.values[:-1]

    for iclass in list_class_rep:
        df_1_class = df_1[df_1['class']==iclass].drop(['class'], axis=1).reset_index(drop=True)
        df_2_class = df_2[df_2['class']==iclass].drop(['class'], axis=1).reset_index(drop=True)

        if not (df_1_class.shape[0] >= image_threshold and df_2_class.shape[0] >= image_threshold):
            continue

        for ifeature in selected_features:
            ax = plt.subplot(1, 1, 1)
            ax.set_xlabel(ifeature + ' (normalized)')
            ax.set_ylabel('Density')

            # feature_1 = df_1_class[ifeature]
            # feature_2 = df_2_class[ifeature]
            # features = [feature_1, feature_2]

            # min_bin = np.min([min(feature_1), min(feature_2)])
            # max_bin = np.max([max(feature_1), max(feature_2)])

            # normalized_features = np.divide((np.array(features, dtype=object) - min_bin), (max_bin - min_bin))

            # if adaptive_bins == 'yes':
            #     # n_bins_feature = int(len(log_feature_1) / 5)
                # n_bins_feature = np.min([20, int(len(log_feature_1) / 5), int(len(log_feature_2) / 3)])
                # n_bins_feature = np.max([5, n_bins_feature])

            # histogram = plt.hist(normalized_features, histtype='stepfilled', bins=n_bins_feature, range=(0, 1), density=True, alpha=0.5, label=data_labels)




            feature_1 = np.divide(df_1_class[ifeature] - df_1_class[ifeature].mean(), df_1_class[ifeature].std()).values
            feature_2 = np.divide(df_2_class[ifeature] - df_1_class[ifeature].mean(), df_1_class[ifeature].std()).values

            log_feature_1, log_feature_2 = np.array([]), np.array([])
            for i in feature_1:
                if i >= 0:
                    log_i = np.emath.logn(10, 1 + i)
                else:
                    log_i = -np.emath.logn(10, 1 - i)
                log_feature_1 = np.append(log_feature_1, log_i)
            for i in feature_2:
                if i >= 0:
                    log_i = np.emath.logn(10, 1 + i)
                else:
                    log_i = -np.emath.logn(10, 1 - i)
                log_feature_2 = np.append(log_feature_2, log_i)

            # log_feature_1 = np.tanh(feature_1)
            # log_feature_2 = np.tanh(feature_2)

            features = [log_feature_1, log_feature_2]

            min_bin = np.min([min(log_feature_1), min(log_feature_2)])
            max_bin = np.max([max(log_feature_1), max(log_feature_2)])

            normalized_features = np.divide((np.array(features, dtype=object) - min_bin), (max_bin - min_bin))

            if adaptive_bins == 'yes':
                # n_bins_feature = int(len(log_feature_1) / 5)
                n_bins_feature = np.min([20, int(len(log_feature_1) / 5), int(len(log_feature_2) / 3)])
                n_bins_feature = np.max([5, n_bins_feature])

            histogram = plt.hist(normalized_features, histtype='stepfilled', bins=n_bins_feature, range=(0, 1), density=True, alpha=0.5, label=data_labels)




            density_1 = histogram[0][0]
            density_2 = histogram[0][1]

            if distance_type == 'Hellinger':
                distance = dm.HellingerDistance(density_1, density_2)
            elif distance_type == 'Wasserstein':
                distance = dm.WassersteinDistance(density_1, density_2)
            elif distance_type == 'KL':
                distance = dm.KullbackLeibler(density_1, density_2)
            elif distance_type == 'Theta':
                distance = dm.ThetaDistance(density_1, density_2)
            elif distance_type == 'Chi':
                distance = dm.ChiDistance(density_1, density_2)
            elif distance_type == 'I':
                distance = dm.IDistance(density_1, density_2)
            elif distance_type == 'Imax':
                distance = dm.ImaxDistance(density_1, density_2)

            plt.title(distance_type + ' Distance = %.3f' % distance)
            plt.legend(loc=1)
            plt.tight_layout()

            outpath_feature = outpath + ifeature + '/'
            Path(outpath_feature).mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath_feature + iclass + '_' + distance_type + '.png')
            plt.close()
            ax.clear()


def PlotGlobalDistanceversusBin_feature(feature_files, outpath, image_threshold, distance_type):
            
    print('-----------------Now plotting global distances of feature v.s. numbers of bin.-----------------')

    df_1 = pd.read_csv(feature_files[0], index_col=0)
    df_2 = pd.read_csv(feature_files[1], index_col=0)

    class_1 = df_1['class'].to_list()
    class_2 = df_2['class'].to_list()
    df_count_1 = pd.DataFrame(np.unique(class_1, return_counts=True)).transpose()
    df_count_2 = pd.DataFrame(np.unique(class_2, return_counts=True)).transpose()
    list_class_1 = df_count_1[df_count_1.iloc[:, 1]>=image_threshold].iloc[:, 0].to_list()
    list_class_2 = df_count_2[df_count_2.iloc[:, 1]>=image_threshold].iloc[:, 0].to_list()
    list_class_rep = list(set(list_class_1) & set(list_class_2))
    list.sort(list_class_rep)

    list_features = df_1.columns.to_list()[:-1]
    list_n_bins = [5, 10, 15, 20, 25, 30]

    df_global_distance = pd.DataFrame(columns=[in_bins for in_bins in list_n_bins], index=[iclass for iclass in list_class_rep])
    for in_bins in list_n_bins:
        list_global_distance = []
        for iclass in list_class_rep:
            df_1_class = df_1[df_1['class']==iclass].drop(['class'], axis=1).reset_index(drop=True)
            df_2_class = df_2[df_2['class']==iclass].drop(['class'], axis=1).reset_index(drop=True)

            list_distance = []
            for ifeature in list_features:
                feature_1 = df_1_class[ifeature]
                feature_2 = df_2_class[ifeature]
                features = [feature_1, feature_2]

                min_bin = np.min([min(feature_1), min(feature_2)])
                max_bin = np.max([max(feature_1), max(feature_2)])

                normalized_features = np.divide((np.array(features, dtype=object) - min_bin), (max_bin - min_bin))

                histogram_1 = np.histogram(normalized_features[0], bins=in_bins, range=(0, 1), density=True)
                histogram_2 = np.histogram(normalized_features[1], bins=in_bins, range=(0, 1), density=True)
                density_1 = histogram_1[0]
                density_2 = histogram_2[0]

                if distance_type == 'Hellinger':
                    distance = dm.HellingerDistance(density_1, density_2)
                elif distance_type == 'Wasserstein':
                    distance = dm.WassersteinDistance(density_1, density_2)
                elif distance_type == 'KL':
                    distance = dm.KullbackLeibler(density_1, density_2)
                elif distance_type == 'Theta':
                    distance = dm.ThetaDistance(density_1, density_2)
                elif distance_type == 'Chi':
                    distance = dm.ChiDistance(density_1, density_2)
                elif distance_type == 'I':
                    distance = dm.IDistance(density_1, density_2)
                elif distance_type == 'Imax':
                    distance = dm.ImaxDistance(density_1, density_2)
                list_distance.append(distance)

            global_distance_each_class = np.average(list_distance)
            list_global_distance.append(global_distance_each_class)

        df_global_distance[in_bins] = list_global_distance

    df_global_distance = df_global_distance.transpose()

    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('Number of bins')
    ax.set_ylabel(distance_type + ' Distance')
    plt.figure(figsize=(10, 10))
    
    for iclass in list_class_rep:
        plt.plot(list_n_bins, df_global_distance[iclass], label=iclass)

    plt.legend(bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    plt.tight_layout()

    Path(outpath).mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath + distance_type + '_distance_bin_feature.png')
    plt.close()
    ax.clear()


def GlobalDistance_feature(feature_files, outpath, n_bins_feature, adaptive_bins, image_threshold, distance_type):

    print('-----------------Now computing global distances on feature (threshold: {}, distance: {}).-----------------'.format(image_threshold, distance_type))

    df_1 = pd.read_csv(feature_files[0], index_col=0)
    df_2 = pd.read_csv(feature_files[1], index_col=0)

    list_class_1 = np.unique(df_1['class'])
    list_class_2 = np.unique(df_2['class'])
    list_class_rep = list(set(list_class_1) & set(list_class_2))
    list.sort(list_class_rep)

    list_features = df_1.columns.to_list()[:-1]

    df = pd.DataFrame()
    list_global_distance = []
    for iclass in list_class_rep:
        df_1_class = df_1[df_1['class']==iclass].drop(['class'], axis=1).reset_index(drop=True)
        df_2_class = df_2[df_2['class']==iclass].drop(['class'], axis=1).reset_index(drop=True)

        if not (df_1_class.shape[0] >= image_threshold and df_2_class.shape[0] >= image_threshold):
            continue

        list_distance = []
        for ifeature in list_features:
            # feature_1 = df_1_class[ifeature]
            # feature_2 = df_2_class[ifeature]
            # features = [feature_1, feature_2]

            # min_bin = np.min([min(feature_1), min(feature_2)])
            # max_bin = np.max([max(feature_1), max(feature_2)])

            # normalized_features = np.divide((np.array(features, dtype=object) - min_bin), (max_bin - min_bin))

            # if adaptive_bins == 'yes':
            #     # n_bins_feature = int(len(log_feature_1) / 5)
                # n_bins_feature = np.min([20, int(len(log_feature_1) / 5), int(len(log_feature_2) / 3)])
                # n_bins_feature = np.max([5, n_bins_feature])

            # histogram_1 = np.histogram(normalized_features[0], bins=n_bins_feature, range=(0, 1), density=True)
            # histogram_2 = np.histogram(normalized_features[1], bins=n_bins_feature, range=(0, 1), density=True)
            
            
            

            feature_1 = np.divide(df_1_class[ifeature] - df_1_class[ifeature].mean(), df_1_class[ifeature].std()).values
            feature_2 = np.divide(df_2_class[ifeature] - df_1_class[ifeature].mean(), df_1_class[ifeature].std()).values

            log_feature_1, log_feature_2 = np.array([]), np.array([])
            for i in feature_1:
                if i >= 0:
                    log_i = np.emath.logn(10, 1 + i)
                else:
                    log_i = -np.emath.logn(10, 1 - i)
                log_feature_1 = np.append(log_feature_1, log_i)
            for i in feature_2:
                if i >= 0:
                    log_i = np.emath.logn(10, 1 + i)
                else:
                    log_i = -np.emath.logn(10, 1 - i)
                log_feature_2 = np.append(log_feature_2, log_i)

            # log_feature_1 = np.tanh(feature_1)
            # log_feature_2 = np.tanh(feature_2)

            features = [log_feature_1, log_feature_2]

            min_bin = np.min([min(log_feature_1), min(log_feature_2)])
            max_bin = np.max([max(log_feature_1), max(log_feature_2)])

            normalized_features = np.divide((np.array(features, dtype=object) - min_bin), (max_bin - min_bin))

            if adaptive_bins == 'yes':
                # n_bins_feature = int(len(log_feature_1) / 5)
                n_bins_feature = np.min([20, int(len(log_feature_1) / 5), int(len(log_feature_2) / 3)])
                n_bins_feature = np.max([5, n_bins_feature])

            histogram_1 = np.histogram(normalized_features[0], bins=n_bins_feature, range=(0, 1), density=True)
            histogram_2 = np.histogram(normalized_features[1], bins=n_bins_feature, range=(0, 1), density=True)

            
            
            
            density_1 = histogram_1[0]
            density_2 = histogram_2[0]

            if distance_type == 'Hellinger':
                distance = dm.HellingerDistance(density_1, density_2)
            elif distance_type == 'Wasserstein':
                distance = dm.WassersteinDistance(density_1, density_2)
            elif distance_type == 'KL':
                distance = dm.KullbackLeibler(density_1, density_2)
            elif distance_type == 'Theta':
                distance = dm.ThetaDistance(density_1, density_2)
            elif distance_type == 'Chi':
                distance = dm.ChiDistance(density_1, density_2)
            elif distance_type == 'I':
                distance = dm.IDistance(density_1, density_2)
            elif distance_type == 'Imax':
                distance = dm.ImaxDistance(density_1, density_2)
            list_distance.append(distance)

        df[iclass] = list_distance

        global_distance_each_class = np.average(list_distance)
        list_global_distance.append(global_distance_each_class)

        Path(outpath).mkdir(parents=True, exist_ok=True)
        with open(outpath + 'Global_' + distance_type + '_Distance_feature.txt', 'a') as f:
            f.write('%-20s%-20f\n' % (iclass, global_distance_each_class))

    df = df.transpose()
    df.columns = [i for i in list_features]
    df.to_excel(outpath + distance_type + '_Distance_class_feature.xlsx', index=True)

    global_distance = np.average(list_global_distance)
    with open(outpath + 'Global_' + distance_type + '_Distance_feature.txt', 'a') as f:
        f.write(f'\n Global Distance: {global_distance}\n')


def GlobalDistance_feature_x(feature_files, outpath, n_bins_feature, adaptive_bins, distance_type):

    print('-----------------Now computing global distances on feature (distance: {}).-----------------'.format(distance_type))

    df_1 = pd.read_csv(feature_files[0], index_col=0).drop(['class'], axis=1)
    df_2 = pd.read_csv(feature_files[1], index_col=0).drop(['class'], axis=1)

    list_features = df_1.columns.to_list()

    df = pd.DataFrame()
    list_distance = []
    
    for ifeature in list_features:

        feature_1 = np.divide(df_1[ifeature] - df_1[ifeature].mean(), df_1[ifeature].std()).values
        feature_2 = np.divide(df_2[ifeature] - df_1[ifeature].mean(), df_1[ifeature].std()).values

        log_feature_1, log_feature_2 = np.array([]), np.array([])
        for i in feature_1:
            if i >= 0:
                log_i = np.emath.logn(10, 1 + i)
            else:
                log_i = -np.emath.logn(10, 1 - i)
            log_feature_1 = np.append(log_feature_1, log_i)
        for i in feature_2:
            if i >= 0:
                log_i = np.emath.logn(10, 1 + i)
            else:
                log_i = -np.emath.logn(10, 1 - i)
            log_feature_2 = np.append(log_feature_2, log_i)

        features = [log_feature_1, log_feature_2]

        min_bin = np.min([min(log_feature_1), min(log_feature_2)])
        max_bin = np.max([max(log_feature_1), max(log_feature_2)])

        normalized_features = np.divide((np.array(features, dtype=object) - min_bin), (max_bin - min_bin))

        if adaptive_bins == 'yes':
            n_bins_feature = np.min([20, int(len(log_feature_1) / 5), int(len(log_feature_2) / 3)])
            n_bins_feature = np.max([5, n_bins_feature])

        histogram_1 = np.histogram(normalized_features[0], bins=n_bins_feature, range=(0, 1), density=True)
        histogram_2 = np.histogram(normalized_features[1], bins=n_bins_feature, range=(0, 1), density=True) 
        
        density_1 = histogram_1[0]
        density_2 = histogram_2[0]

        if distance_type == 'Hellinger':
            distance = dm.HellingerDistance(density_1, density_2)
        elif distance_type == 'Wasserstein':
            distance = dm.WassersteinDistance(density_1, density_2)
        elif distance_type == 'KL':
            distance = dm.KullbackLeibler(density_1, density_2)
        elif distance_type == 'Theta':
            distance = dm.ThetaDistance(density_1, density_2)
        elif distance_type == 'Chi':
            distance = dm.ChiDistance(density_1, density_2)
        elif distance_type == 'I':
            distance = dm.IDistance(density_1, density_2)
        elif distance_type == 'Imax':
            distance = dm.ImaxDistance(density_1, density_2)
        list_distance.append(distance)

    df[0] = list_distance

    Path(outpath).mkdir(parents=True, exist_ok=True)

    df = df.transpose()
    df.columns = [i for i in list_features]
    df.to_excel(outpath + distance_type + '_Distance_class_feature_x.xlsx', index=True)

    global_distance = np.average(list_distance)
    with open(outpath + 'Global_' + distance_type + '_Distance_feature_x.txt', 'a') as f:
        f.write(f'\n Global Distance: {global_distance}\n')