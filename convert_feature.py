import os
import argparse
from pathlib import Path

import pandas as pd

from utils_analysis.lib import feature_extraction as fe


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


parser = argparse.ArgumentParser(description='extract the features of a dataset and save them as a file')
parser.add_argument('-datapaths', nargs='*', help='paths of datasets')
parser.add_argument('-dataset_labels', nargs='*', help='label of each dataset')
parser.add_argument('-outpath', help='path for saving output file')
parser.add_argument('-nice_feature', choices=['yes', 'no'], help='only use nice features or not')
args = parser.parse_args()

if __name__ == '__main__':

    for idatapath, ilabel in zip(args.datapaths, args.dataset_labels):
        df_all_feat = ConcatAllClasses(idatapath, args.nice_feature)
        Path(args.outpath).mkdir(parents=True, exist_ok=True)
        df_all_feat.to_csv(args.outpath + ilabel + '_feature.csv')