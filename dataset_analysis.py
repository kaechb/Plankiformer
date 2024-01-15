import argparse

from utils_analysis import feature
from utils_analysis import pixel
from utils_analysis import abundance
from utils_analysis import sampling_date as sdate
from utils_analysis import feature_components as fc
from utils_analysis import pixel_components as pc
from utils_analysis import image_wise_distance as iwd
from utils_analysis import mahalanobis


parser = argparse.ArgumentParser(description='Plot some figures about data distribution')
parser.add_argument('-datapaths', nargs='*', help='path of the dataset')
parser.add_argument('-datapath_labels', nargs='*', help='name of the dataset')
parser.add_argument('-train_datapath', help='path of train dataset')
parser.add_argument('-outpath', help='path of the output')
parser.add_argument('-selected_features', nargs='*', help='select the features that you want to analyse')
parser.add_argument('-selected_pixels', nargs='*', help='select the pixels that you want to analyse')
parser.add_argument('-n_bins_feature', type=int, help='number of bins in the feature distribution plot')
parser.add_argument('-n_bins_pixel', type=int, help='number of bins in the pixel distribution plot')
parser.add_argument('-adaptive_bins', choices=['yes', 'no'], default='no', help='self-adapt the number of bins by the number of images')
parser.add_argument('-resized_length', type=int, default=64, help='length of resized image')

parser.add_argument('-feature_files', nargs='*', help='feature files of all datasets')
parser.add_argument('-pixel_files', nargs='*', help='pixel files of all datasets')
parser.add_argument('-PCA_files_feature', nargs='*', help='principal components file of all datasets')
parser.add_argument('-PCA_files_pixel', nargs='*', help='principal components file of all datasets')
parser.add_argument('-outpath_feature', help='path of the output')
parser.add_argument('-outpath_pixel', help='path of the output')
parser.add_argument('-selected_components_feature', nargs='*', help='select the feature components that you want to analyse')
parser.add_argument('-selected_components_pixel', nargs='*', help='select the pixel components that you want to analyse')
parser.add_argument('-data_labels', nargs='*', help='name of the dataset')
parser.add_argument('-explained_variance_ratio_feature', help='explained variance ratio file of feature components')
parser.add_argument('-explained_variance_ratio_pixel', help='explained variance ratio file of pixel components')
parser.add_argument('-PCA', choices=['yes', 'no'], default='yes', help='apply PCA or not')
parser.add_argument('-feature_or_pixel', choices=['feature', 'pixel', 'both'], default='both', help='analysis on features or pixels')
parser.add_argument('-image_threshold', type=int, help='minimum number of images of each class for calculating Distance')
parser.add_argument('-distance_type', choices=['Hellinger', 'Wasserstein', 'KL', 'Theta', 'Chi', 'I', 'Imax', 'Imagewise', 'Mahalanobis'], help='the type of distribution distance')
parser.add_argument('-mahal_mean', choices=['yes', 'no'], help='type of mahalanobis distance')
parser.add_argument('-global_x', choices=['yes', 'no'], default='no', help='PCA on data over all classes or not')
args = parser.parse_args()


if __name__ == '__main__':

    # # sdate.PlotSamplingDate(args.train_datapath, args.outpath)
    # # sdate.PlotSamplingDateEachClass(args.train_datapath, args.outpath)
    # sdate.PlotSamplingDate_with_test(args.datapaths, args.outpath)
    # # abundance.PlotAbundance(args.datapaths, args.outpath, args.datapath_labels)
    # abundance.PlotAbundanceSep(args.datapaths, args.outpath, args.datapath_labels)
    # abundance.PlotAbundance_overall(args.datapaths, args.outpath, args.datapath_labels)

    if args.global_x == 'no':
        if args.PCA == 'no':
            if args.distance_type == 'Mahalanobis':
                mahalanobis.GlobalDistance_feature(args.feature_files, args.outpath_feature, args.PCA, args.image_threshold, args.mahal_mean)
            elif args.distance_type == 'Imagewise':
                iwd.GlobalDistance_feature(args.feature_files, args.outpath_feature, args.PCA, args.image_threshold)
            else:
                if args.feature_or_pixel == 'both':
                    # feature.PlotFeatureDistribution(args.feature_files, args.outpath_feature, args.selected_features, args.n_bins_feature, args.adaptive_bins, args.data_labels, args.image_threshold, args.distance_type)
                    # feature.PlotGlobalDistanceversusBin_feature(args.feature_files, args.outpath_feature, args.image_threshold, args.distance_type)
                    feature.GlobalDistance_feature(args.feature_files, args.outpath_feature, args.n_bins_feature, args.adaptive_bins, args.image_threshold, args.distance_type)
                    # pixel.PlotPixelDistribution(args.pixel_files, args.outpath_pixel, args.selected_pixels, args.n_bins_pixel, args.data_labels, args.image_threshold, args.distance_type, args.resized_length)
                    # pixel.PlotGlobalDistanceversusBin_pixel(args.pixel_files, args.outpath_pixel, args.image_threshold, args.distance_type, args.resized_length)
                    pixel.GlobalDistance_pixel(args.pixel_files, args.outpath_pixel, args.n_bins_pixel, args.image_threshold, args.distance_type, args.resized_length)
                elif args.feature_or_pixel == 'feature':
                    # feature.PlotFeatureDistribution(args.feature_files, args.outpath_feature, args.selected_features, args.n_bins_feature, args.adaptive_bins, args.data_labels, args.image_threshold, args.distance_type)
                    # feature.PlotGlobalDistanceversusBin_feature(args.feature_files, args.outpath_feature, args.image_threshold, args.distance_type)
                    feature.GlobalDistance_feature(args.feature_files, args.outpath_feature, args.n_bins_feature, args.adaptive_bins, args.image_threshold, args.distance_type)
                elif args.feature_or_pixel == 'pixel':
                    # pixel.PlotPixelDistribution(args.pixel_files, args.outpath_pixel, args.selected_pixels, args.n_bins_pixel, args.data_labels, args.image_threshold, args.distance_type, args.resized_length)
                    # pixel.PlotGlobalDistanceversusBin_pixel(args.pixel_files, args.outpath_pixel, args.image_threshold, args.distance_type, args.resized_length)
                    pixel.GlobalDistance_pixel(args.pixel_files, args.outpath_pixel, args.n_bins_pixel, args.image_threshold, args.distance_type, args.resized_length)
            

        if args.PCA == 'yes':
            if args.distance_type == 'Mahalanobis':
                mahalanobis.GlobalDistance_feature(args.PCA_files_feature, args.outpath_feature, args.PCA, args.image_threshold, args.mahal_mean)
            elif args.distance_type == 'Imagewise':
                iwd.GlobalDistance_feature(args.PCA_files_feature, args.outpath_feature, args.PCA, args.image_threshold)
            else:
                if args.feature_or_pixel == 'both':
                    # fc.PlotFeatureDistribution_PCA(args.PCA_files_feature, args.outpath_feature, args.selected_components_feature, args.n_bins_feature, args.adaptive_bins, args.data_labels, args.image_threshold, args.distance_type)
                    # fc.PlotGlobalDistanceversusBin_feature_PCA(args.PCA_files_feature, args.outpath_feature, args.explained_variance_ratio_feature, args.image_threshold, args.distance_type)
                    fc.GlobalDistance_feature(args.PCA_files_feature, args.outpath_feature, args.n_bins_feature, args.adaptive_bins, args.explained_variance_ratio_feature, args.image_threshold, args.distance_type)
                    # pc.PlotPixelDistribution_PCA(args.PCA_files_pixel, args.outpath_pixel, args.selected_components_pixel, args.n_bins_pixel, args.data_labels, args.image_threshold, args.distance_type)
                    # pc.PlotGlobalDistanceversusBin_pixel_PCA(args.PCA_files_pixel, args.outpath_pixel, args.explained_variance_ratio_pixel, args.image_threshold, args.distance_type)
                    pc.GlobalDistance_pixel(args.PCA_files_pixel, args.outpath_pixel, args.n_bins_pixel, args.explained_variance_ratio_pixel, args.image_threshold, args.distance_type)

                elif args.feature_or_pixel == 'feature':
                    # fc.PlotFeatureDistribution_PCA(args.PCA_files_feature, args.outpath_feature, args.selected_components_feature, args.n_bins_feature, args.adaptive_bins, args.data_labels, args.image_threshold, args.distance_type)
                    # fc.PlotGlobalDistanceversusBin_feature_PCA(args.PCA_files_feature, args.outpath_feature, args.explained_variance_ratio_feature, args.image_threshold, args.distance_type)
                    fc.GlobalDistance_feature(args.PCA_files_feature, args.outpath_feature, args.n_bins_feature, args.adaptive_bins, args.explained_variance_ratio_feature, args.image_threshold, args.distance_type)

                elif args.feature_or_pixel == 'pixel':
                    # pc.PlotPixelDistribution_PCA(args.PCA_files_pixel, args.outpath_pixel, args.selected_components_pixel, args.n_bins_pixel, args.data_labels, args.image_threshold, args.distance_type)
                    # pc.PlotGlobalDistanceversusBin_pixel_PCA(args.PCA_files_pixel, args.outpath_pixel, args.explained_variance_ratio_pixel, args.image_threshold, args.distance_type)
                    pc.GlobalDistance_pixel(args.PCA_files_pixel, args.outpath_pixel, args.n_bins_pixel, args.explained_variance_ratio_pixel, args.image_threshold, args.distance_type)

    elif args.global_x == 'yes':
        if args.PCA == 'no':
            feature.GlobalDistance_feature_x(args.feature_files, args.outpath_feature, args.n_bins_feature, args.adaptive_bins, args.distance_type)
        if args.PCA == 'yes':
            fc.GlobalDistance_feature_x(args.PCA_files_feature, args.outpath_feature, args.n_bins_feature, args.adaptive_bins, args.explained_variance_ratio_feature, args.distance_type)