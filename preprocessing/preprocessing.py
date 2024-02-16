# -*- coding: utf-8 -*-
import argparse
import json
import os
import subprocess

import numpy as np
from scipy.ndimage import label

os.environ['LD_LIBRARY_PATH'] = "/home/felix/VTK-8.2.0/_install/lib:/home/felix/InsightToolkit-4.13.3/_install/lib"
clitk_tools = "/home/felix/clitk_private/_install/bin"

parser = argparse.ArgumentParser(description='create_json')
parser.add_argument('--path', default='./path/to/your/dataset', type=str)
parser.add_argument('--normalization', default='range-std', type=str)


def main(path, normalization):
    print("calculate median spacing")
    median_spacing = get_db_median_spacing(path)

    print("resample images")
    resample(path, path, median_spacing)
    print("resample labels")
    resample(path, path, median_spacing, label=True)

    if normalization == "range-std":
        print("convert to float")
        tofloat(path, path)
        print("gaussian normalization")
        range_std_normalization(path, path)

    else:
        print('min/max normalization')
        min_max_normalization(path, path)

    create_json(path)


def getSpacing(clitk_tools, input_image):
    clitkImageInfo = [os.path.join(clitk_tools, 'clitkImageInfo')]
    args_list = clitkImageInfo + [input_image]
    p0 = subprocess.Popen(args_list, stdout=subprocess.PIPE)
    output = p0.communicate()[0]
    output = output.decode()
    output_splitted = output.split(' ')
    output_splitted = output_splitted[3].split('x')
    return [float(output_splitted[0]), float(output_splitted[1]), float(output_splitted[2])]


def get_db_median_spacing(input_db):
    path_images = os.path.join(input_db, "imagesTr")
    spacing_images = []
    for image in os.listdir(path_images):
        image_input_dir = os.path.join(path_images, image)
        spacing_images.append(getSpacing(clitk_tools, image_input_dir))

    # converting list to array
    spacing_images = np.asarray(spacing_images)
    median_spacing = np.median(spacing_images, axis=0)
    return median_spacing


def resample(input_db, output_db, spacing, label=False):
    if not label:
        input_images = os.path.join(input_db, "imagesTr")
        output_images = os.path.join(output_db, "imagesTr")
    else:
        input_images = os.path.join(input_db, "labelsTr")
        output_images = os.path.join(output_db, "labelsTr")

    for image in os.listdir(input_images):
        # Images IRM
        clitkAffineTransform = [os.path.join(clitk_tools, 'clitkAffineTransform')]
        input_list = ['-i', os.path.join(input_images, image)]
        output_list = ['-o', os.path.join(output_images, image),
                       '--spacing=' +
                       str(spacing[0]) + ',' + str(spacing[1]) + ',' + str(spacing[2]),
                       '--adaptive']
        if label:
            output_list.append('--interp=0')
        args_list = clitkAffineTransform + input_list + output_list
        subprocess.run(args_list)
    return None


def resampleToFixSize(input_db, output_db, value, label=False):
    if not label:
        input_images = os.path.join(input_db, "imagesTr")
        output_images = os.path.join(output_db, "imagesTr")
    else:
        input_images = os.path.join(input_db, "labelsTr")
        output_images = os.path.join(output_db, "labelsTr")
    for image in os.listdir(input_images):
        # Images IRM
        spacing = getSpacing(clitk_tools, os.path.join(input_images, image))
        size = getPixelSize(clitk_tools, os.path.join(input_images, image))
        clitkAffineTransform = [os.path.join(clitk_tools, 'clitkAffineTransform')]
        input_list = ['-i', os.path.join(input_images, image)]
        output_list = ['-o', os.path.join(output_images, image),
                       '--spacing=' +
                       str(spacing[0] * size[0] / value) + ',' + str(spacing[1] * size[1] / value) + ',' + str(
                           spacing[2] * size[2] / value), '--adaptive']
        if label:
            output_list.append('--interp=0')
        args_list = clitkAffineTransform + input_list + output_list
        subprocess.run(args_list)
    return None


def min_max_normalization(input_db, output_db):
    input_images = os.path.join(input_db, "imagesTr")
    output_images = os.path.join(output_db, "imagesTr")
    for image in os.listdir(input_images):
        input_list = ['-i', os.path.join(input_images, image)]
        output_list = ['-o', os.path.join(output_images, image)]
        clitkNormalizeImage = [os.path.join(clitk_tools, 'clitkNormalizeImageFilter')]
        args_list = clitkNormalizeImage + input_list + output_list
        subprocess.run(args_list)
    return None


def range_std_normalization(input_db, output_db):
    input_images = os.path.join(input_db, "pred")
    output_images = os.path.join(output_db, "pred")
    for image in os.listdir(input_images):
        stats = getImageStatistics(clitk_tools, os.path.join(input_images, image))
        input_list = ['-i', os.path.join(input_images, image)]
        output_list = ['-o', os.path.join(output_images, image)]
        operation_1 = ['-s', str(stats[0]), '-t', str(0)]

        clitkImageArithm = [os.path.join(clitk_tools, 'clitkImageArithm')]
        args_list = clitkImageArithm + input_list + output_list + operation_1
        subprocess.run(args_list)
        input_list = ['-i', os.path.join(output_images, image)]
        operation_2 = ['-s', str(stats[1]), '-t', str(1)]
        args_list = clitkImageArithm + input_list + output_list + operation_2
        subprocess.run(args_list)
    return None


def tofloat(input_db, output_db, type='float'):
    input_images = os.path.join(input_db, "pred")
    output_images = os.path.join(output_db, "pred")
    for image in os.listdir(input_images):
        input_list = ['-i', os.path.join(input_images, image)]
        output_list = ['-o', os.path.join(output_images, image)]
        operation = ['-t', type]
        clitkImageArithm = [os.path.join(clitk_tools, 'clitkImageConvert')]
        args_list = clitkImageArithm + input_list + output_list + operation
        subprocess.run(args_list)
    return None


def getImageStatistics(clitk_tools, input_image):
    clitkImageStatistics = [os.path.join(clitk_tools, 'clitkImageStatistics')]
    args_list = clitkImageStatistics + ['-v', input_image]
    p0 = subprocess.Popen(args_list, stdout=subprocess.PIPE)
    output = p0.communicate()[0]
    output = output.decode()
    output_splitted = output.split('\n')
    mean = float(output_splitted[6][5:])
    std = 1 / float(output_splitted[9][6:])
    return [-mean, std]


def getPixelSize(clitk_tools, input_image):
    clitkImageInfo = [os.path.join(clitk_tools, 'clitkImageInfo')]
    args_list = clitkImageInfo + [input_image]
    p0 = subprocess.Popen(args_list, stdout=subprocess.PIPE)
    output = p0.communicate()[0]
    output = output.decode()
    output_splitted = output.split(' ')
    output_splitted = output_splitted[2].split('x')
    return [float(output_splitted[0]), float(output_splitted[1]), float(output_splitted[2])]


def create_json(path):
    # Define the directories
    images_dir = 'imagesTr'
    labels_dir = 'labelsTr'

    # Get the list of image and label files
    images_files = sorted([f for f in os.listdir(os.path.join(path, images_dir)) if f.endswith('.nii.gz')])
    labels_files = sorted([f for f in os.listdir(os.path.join(path, labels_dir)) if f.endswith('.nii.gz')])

    # Check if the number of image files and label files are the same
    if len(images_files) != len(labels_files):
        raise ValueError('The number of image and label files does not match.')

    # Create the training data list
    training_data = []
    for img, lbl in zip(images_files, labels_files):
        training_data.append({
            "image": os.path.join(images_dir, img),
            "label": os.path.join(labels_dir, lbl)
        })

    # Define the dictionary to be converted to json
    data_to_write = {
        "training": training_data
    }

    # Define the json file name
    json_file_name = f'{path}/dataset.json'

    # Write the dictionary to a json file
    with open(json_file_name, 'w') as json_file:
        json.dump(data_to_write, json_file, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.path, args.normalization)
