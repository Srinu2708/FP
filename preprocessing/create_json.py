import argparse
import json
import os

# Assuming 'imagesTr' and 'labelsTr' are the given directories and they contain the same number of files.
# Also assuming that the files are named in a way that after sorting the list of files, each image corresponds to the label with the same index.

parser = argparse.ArgumentParser(description='create_json')
parser.add_argument('--path', default='./path/to/your/dataset', type=str)


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
    create_json(args.path)
