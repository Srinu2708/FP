# -*- coding: utf-8 -*-
import argparse
import json
import os

import SimpleITK as sitk
import numpy as np

parser = argparse.ArgumentParser(description='create_json')
parser.add_argument('--path', default='./path/to/your/dataset', type=str)
parser.add_argument('--normalization', default='range-std', type=str)


def main(path, normalization):
    print("calculate median spacing")
    median_spacing = get_db_median_spacing(path)
    print("resample images")
    resample(path, median_spacing)

    if normalization == "range-std":
        print("range-std normalization")
        range_std_normalization(path)

    else:
        print('min/max normalization')
        min_max_normalization(path)

    create_json(path)


def get_db_median_spacing(path):
    # Path to the subdirectory containing the images
    images_path = os.path.join(path, "imagesTr")

    # Check if the path exists
    if not os.path.exists(images_path):
        raise ValueError(f"The directory {images_path} does not exist.")

    # List to store spacings
    spacings = []

    # Traverse the directory to find .nii.gz files
    for file in os.listdir(images_path):
        if file.endswith(".nii.gz"):
            # Construct the full path to the file
            file_path = os.path.join(images_path, file)

            # Read the image
            image = sitk.ReadImage(file_path)

            # Add the spacing to the list
            spacings.append(image.GetSpacing())

    # Check if any images were found
    if not spacings:
        raise ValueError("No .nii.gz images found.")

    # Calculate median spacing for each dimension
    spacings = np.array(spacings)
    median_spacing = np.median(spacings, axis=0)

    return median_spacing


def resample(directory, new_spacing):
    for subdirectory in ['imagesTr', 'labelsTr']:
        subdirectory_path = os.path.join(directory, subdirectory)

        # Check if the subdirectory exists
        if not os.path.exists(subdirectory_path):
            print(f"Subdirectory {subdirectory_path} does not exist. Skipping.")
            continue

        for file in os.listdir(subdirectory_path):
            if file.endswith(".nii.gz"):
                file_path = os.path.join(subdirectory_path, file)
                image = sitk.ReadImage(file_path)

                # Get the original spacing, size, and origin
                original_spacing = image.GetSpacing()
                original_size = image.GetSize()
                original_origin = image.GetOrigin()

                # Calculate the new size
                new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in
                            zip(original_size, original_spacing, new_spacing)]

                # Resample the image
                resampler = sitk.ResampleImageFilter()
                resampler.SetSize(new_size)
                resampler.SetOutputSpacing(new_spacing)
                resampler.SetOutputOrigin(original_origin)  # Set the origin to the original origin
                resampler.SetTransform(sitk.Transform())
                resampler.SetInterpolator(sitk.sitkLinear)  # For images
                if subdirectory == 'labelTr':
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # For labels

                resampled_image = resampler.Execute(image)

                # Save the resampled image
                resampled_file_path = os.path.join(subdirectory_path, file)
                sitk.WriteImage(resampled_image, resampled_file_path)

                print(f"Resampled and saved: {resampled_file_path}")


def range_std_normalization(directory):
    images_tr_path = os.path.join(directory, "imagesTr")

    # Check if the imagesTr directory exists
    if not os.path.exists(images_tr_path):
        raise ValueError(f"The directory {images_tr_path} does not exist.")

    for file in os.listdir(images_tr_path):
        if file.endswith(".nii.gz"):
            file_path = os.path.join(images_tr_path, file)
            image = sitk.ReadImage(file_path)

            # Convert the image to float32
            image = sitk.Cast(image, sitk.sitkFloat32)

            # Compute the mean and standard deviation
            stats = sitk.StatisticsImageFilter()
            stats.Execute(image)
            mean = stats.GetMean()
            std_dev = stats.GetSigma()

            # Subtract the mean and divide by the standard deviation
            image_array = sitk.GetArrayFromImage(image)
            normalized_array = (image_array - mean) / std_dev

            # Convert the numpy array back to a SimpleITK Image
            normalized_image = sitk.GetImageFromArray(normalized_array)
            normalized_image.CopyInformation(image)

            # Save the normalized image
            normalized_file_path = os.path.join(images_tr_path, file)
            sitk.WriteImage(normalized_image, normalized_file_path)

            print(f"Normalized and saved: {normalized_file_path}")


def min_max_normalization(directory):
    images_tr_path = os.path.join(directory, "imagesTr")

    # Check if the imagesTr directory exists
    if not os.path.exists(images_tr_path):
        raise ValueError(f"The directory {images_tr_path} does not exist.")

    for file in os.listdir(images_tr_path):
        if file.endswith(".nii.gz"):
            file_path = os.path.join(images_tr_path, file)
            image = sitk.ReadImage(file_path)

            # Convert the image to a numpy array
            image_array = sitk.GetArrayFromImage(image)

            # Compute the min and max
            min_val = np.min(image_array)
            max_val = np.max(image_array)

            # Apply min-max normalization
            normalized_array = (image_array - min_val) / (max_val - min_val)

            # Convert the numpy array back to a SimpleITK Image
            normalized_image = sitk.GetImageFromArray(normalized_array)
            normalized_image.CopyInformation(image)

            # Save the normalized image
            normalized_file_path = os.path.join(images_tr_path, file)
            sitk.WriteImage(normalized_image, normalized_file_path)

            print(f"Normalized and saved: {normalized_file_path}")


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
