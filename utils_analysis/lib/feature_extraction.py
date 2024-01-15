import os
import sys

import pandas as pd
import numpy as np

import cv2
from PIL import Image


def ResizeWithProportions(im, desired_size):
    """
    Take and image and resize it to a square of the desired size.
    0) If any dimension of the image is larger than the desired size, shrink until the image can fully fit in the
    desired size
    1) Add black paddings to create a square
    """

    old_size = im.size
    largest_dim = max(old_size)
    smallest_dim = min(old_size)

    # If the image dimensions are very different, reducing the larger one to `desired_size` can make the other
    # dimension too small. We impose that it be at least 4 pixels.
    if desired_size * smallest_dim / largest_dim < 4:
        print('Image size: ({},{})'.format(largest_dim, smallest_dim))
        print('Desired size: ({},{})'.format(desired_size, desired_size))
        raise ValueError(
            'Images are too extreme rectangles to be reduced to this size. Try increasing the desired image size.')

    rescaled = 0  # This flag tells us whether there was a rescaling of the image (besides the padding).
    # We can use it as feature for training.

    # 0) If any dimension of the image is larger than the desired size, shrink until the image can fully fit
    # in the desired size
    if max(im.size) > desired_size:
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        # print('new_size:',new_size)
        sys.stdout.flush()
        im = im.resize(new_size, Image.LANCZOS)
        rescaled = 1

    # 1) Add black paddings to create a square
    new_im = Image.new("RGB", (desired_size, desired_size), color=0)
    new_im.paste(im, ((desired_size - im.size[0]) // 2,
                      (desired_size - im.size[1]) // 2))

    return new_im, rescaled

def LoadExtraFeatures(class_image_datapath, nice_feature):
    df_extra_feat = pd.DataFrame()
    if nice_feature == 'no':
        # dfFeatExtra1 = pd.DataFrame(columns=['width', 'height', 'w_rot', 'h_rot', 'angle_rot', 'aspect_ratio_2',
        #                                     'rect_area', 'contour_area', 'contour_perimeter', 'extent',
        #                                     'compactness', 'formfactor', 'hull_area', 'solidity_2', 'hull_perimeter',
        #                                     'ESD', 'Major_Axis', 'Minor_Axis', 'Angle', 'Eccentricity1', 'Eccentricity2',
        #                                     'Convexity', 'Roundness', 'blurriness', 'noise'])
        dfFeatExtra1 = pd.DataFrame(columns=['width', 'height', 'w_rot', 'h_rot', 'angle_rot', 'aspect_ratio',
                                            'rect_area', 'contour_area', 'contour_perimeter', 'extent',
                                            'compactness', 'formfactor', 'hull_area', 'solidity', 'hull_perimeter',
                                            'ESD', 'major_axis', 'minor_axis', 'angle', 'eccentricity',
                                            'convexity', 'roundness', 'intensity_R_mean', 'intensity_G_mean', 'intensity_B_mean',
                                            'intensity_R_std', 'intensity_G_std', 'intensity_B_std',
                                            'hue_mean', 'saturation_mean', 'brightness_mean',
                                            'hue_std', 'saturation_std', 'brightness_std', 'blurriness', 'noise'])
        dfFeatExtra2 = pd.DataFrame(columns=['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03',
                                            'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'nu20', 'nu11',
                                            'nu02', 'nu30', 'nu21', 'nu12', 'nu03'])
        dfFeatExtra3 = pd.DataFrame(columns=['hu0', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6'])
    elif nice_feature == 'yes':
        # dfFeatExtra1 = pd.DataFrame(columns=['width', 'height', 'w_rot', 'aspect_ratio_2', 'extent',
        #                                     'formfactor', 'solidity_2', 'Angle',
        #                                     'Convexity', 'Roundness', 'blurriness', 'noise'])
        dfFeatExtra1 = pd.DataFrame(columns=['width', 'height', 'w_rot', 'h_rot', 'aspect_ratio_2',
                                            'rect_area', 'contour_perimeter', 'extent',
                                            'compactness', 'formfactor', 'solidity_2', 'hull_perimeter',
                                            'ESD', 'Angle', 'Eccentricity1', 'Eccentricity2',
                                            'Convexity', 'Roundness', 'blurriness', 'noise'])
        dfFeatExtra2 = pd.DataFrame(columns=['mu03', 'nu20', 'nu02', 'nu30', 'nu03'])

    list_image = os.listdir(class_image_datapath)
    for img in list_image:
        if img == 'Thumbs.db':
            continue
        
        # image = cv2.imread(class_image_datapath + img)
        # image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)

        # resize by PIL
        image = Image.open(class_image_datapath + img)
        image, rescaled = ResizeWithProportions(image, 224)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (2, 2))  # blur the image
        ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Find the largest contour
        cnt = max(contours, key=cv2.contourArea)
        # Bounding rectangle
        x, y, width, height = cv2.boundingRect(cnt)
        # Rotated rectangle
        rot_rect = cv2.minAreaRect(cnt)
        rot_box = cv2.boxPoints(rot_rect)
        rot_box = np.int0(rot_box)
        w_rot = rot_rect[1][0]
        h_rot = rot_rect[1][1]
        angle_rot = rot_rect[2]
        # Find Image moment of largest contour
        M = cv2.moments(cnt)
        H = cv2.HuMoments(M)
        hu00, hu01, hu02, hu03, hu04, hu05, hu06 = H[0][0], H[1][0], H[2][0], H[3][0], H[4][0], H[5][0], H[6][0]
        # Find centroid
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # Find the Aspect ratio or elongation --It is the ratio of width to height of bounding rect of the object.
        aspect_ratio = float(width) / height
        # Rectangular area
        rect_area = width * height
        # Area of the contour
        contour_area = cv2.contourArea(cnt)
        # Perimeter of the contour
        contour_perimeter = cv2.arcLength(cnt, True)
        # Extent --Extent is the ratio of contour area to bounding rectangle area
        extent = float(contour_area) / rect_area
        # Compactness -- from MATLAB
        compactness = (np.square(contour_perimeter)) / (4 * np.pi * contour_area)
        # Form factor
        formfactor = (4 * np.pi * contour_area) / (np.square(contour_perimeter))
        # Convex hull points
        hull_2 = cv2.convexHull(cnt)
        # Convex Hull Area
        hull_area = cv2.contourArea(hull_2)
        # solidity --Solidity is the ratio of contour area to its convex hull area.
        solidity = float(contour_area) / hull_area
        # Hull perimeter
        hull_perimeter = cv2.arcLength(hull_2, True)
        # Equivalent circular Diameter-is the diameter of the circle whose area is same as the contour area.
        ESD = np.sqrt(4 * contour_area / np.pi)
        # Orientation, Major Axis, Minos axis -Orientation is the angle at which object is directed
        (x1, y1), (Major_Axis, Minor_Axis), angle = cv2.fitEllipse(cnt)
        # Eccentricity or ellipticity.
        Eccentricity1 = Minor_Axis / Major_Axis
        Mu02 = M['m02'] - (cy * M['m01'])
        Mu20 = M['m20'] - (cx * M['m10'])
        Mu11 = M['m11'] - (cx * M['m01'])
        Eccentricity2 = (np.square(Mu02 - Mu20)) + 4 * Mu11 / contour_area
        # Convexity
        Convexity = hull_perimeter / contour_perimeter
        # Roundness
        Roundness = (4 * np.pi * contour_area) / (np.square(hull_perimeter))

        BGR_mean_std = cv2.meanStdDev(image, mask=thresh)
        HSV_mean_std = cv2.meanStdDev(HSV, mask=thresh)
        B_mean = BGR_mean_std[0][0][0]
        G_mean = BGR_mean_std[0][1][0]
        R_mean = BGR_mean_std[0][2][0]
        B_std = BGR_mean_std[1][0][0]
        G_std = BGR_mean_std[1][1][0]
        R_std = BGR_mean_std[1][2][0]
        hue_mean = HSV_mean_std[0][0][0]
        saturation_mean = HSV_mean_std[0][1][0]
        brightness_mean = HSV_mean_std[0][2][0]
        hue_std = HSV_mean_std[1][0][0]
        saturation_std = HSV_mean_std[1][1][0]
        brightness_std = HSV_mean_std[1][2][0]

        # blurriness
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        blurriness = np.mean(np.abs(laplacian))
        # noise
        noise = np.std(image)

        if nice_feature == 'no':
            # dfFeatExtra1.loc[img] = [width, height, w_rot, h_rot, angle_rot, aspect_ratio, rect_area, contour_area,
            #                         contour_perimeter, extent, compactness, formfactor, hull_area, solidity,
            #                         hull_perimeter, ESD, Major_Axis, Minor_Axis, angle, Eccentricity1,
            #                         Eccentricity2, Convexity, Roundness, blurriness, noise]
            dfFeatExtra1.loc[img] = [width, height, w_rot, h_rot, angle_rot, aspect_ratio, 
                                    rect_area, contour_area, contour_perimeter, extent,
                                    compactness, formfactor, hull_area, solidity, hull_perimeter, 
                                    ESD, Major_Axis, Minor_Axis, angle, Eccentricity1,
                                    Convexity, Roundness, R_mean, G_mean, B_mean, 
                                    R_std, G_std, B_std, 
                                    hue_mean, saturation_mean, brightness_mean,
                                    hue_std, saturation_std, brightness_std, blurriness, noise]
            dfFeatExtra2.loc[img] = M
            dfFeatExtra3.loc[img] = [hu00, hu01, hu02, hu03, hu04, hu05, hu06]
        elif nice_feature == 'yes':
            # dfFeatExtra1.loc[img] = [width, height, w_rot, aspect_ratio,
            #                         extent, formfactor, solidity,
            #                         angle, Convexity, Roundness, blurriness, noise]
            dfFeatExtra1.loc[img] = [width, height, w_rot, h_rot, aspect_ratio, rect_area,
                                    contour_perimeter, extent, compactness, formfactor, solidity,
                                    hull_perimeter, ESD, angle, Eccentricity1,
                                    Eccentricity2, Convexity, Roundness, blurriness, noise]
            dfFeatExtra2.loc[img] = {key: M[key] for key in ['mu03', 'nu20', 'nu02', 'nu30', 'nu03']}
        
        
    if nice_feature == 'no':
        df_extra_feat = pd.concat([dfFeatExtra1, dfFeatExtra2, dfFeatExtra3], axis=1) # dataframe of extra features
    elif nice_feature == 'yes':
        # df_extra_feat = dfFeatExtra1
        df_extra_feat = pd.concat([dfFeatExtra1, dfFeatExtra2], axis=1)

    df_extra_feat = df_extra_feat.sort_index() # sort the images by index (filename)

    return df_extra_feat

def ConcatAllFeatures(class_datapath, nice_feature):
    if os.path.exists(class_datapath + 'training_data/'):
        class_image_datapath = class_datapath + 'training_data/' # folder with images inside
        df_feat = pd.read_csv(class_datapath + 'features.tsv', sep='\t') # dataframe of original features 

        # sort the dataframe of original features by image name
        for i in range(df_feat.shape[0]):
            df_feat.loc[i, 'url'] = df_feat.loc[i, 'url'][13:]
        df_feat = df_feat.sort_values(by='url')
        df_feat = df_feat.reset_index(drop=True)
        # df_feat.to_csv(class_datapath + 'features_sorted.tsv') # save sorted original features
        
        # load extra features from image
        df_extra_feat = LoadExtraFeatures(class_image_datapath, nice_feature)
        df_extra_feat = df_extra_feat.reset_index(drop=True)
        # df_extra_feat.to_csv(class_datapath + 'extra_features.tsv') # save extra features

        # original_features = df_feat.columns.to_list()
        # extra_features = df_extra_feat.columns.to_list()
        # all_features = original_features + extra_features
        # df_all_feat = pd.DataFrame(columns=all_features)
        
        df_all_feat = pd.concat([df_feat, df_extra_feat], axis=1) # concatenate orginal and extra features
    
    else:
        class_image_datapath = class_datapath
        df_extra_feat = LoadExtraFeatures(class_image_datapath, nice_feature)
        df_extra_feat = df_extra_feat.reset_index(drop=True)
        df_all_feat = df_extra_feat

    return df_all_feat