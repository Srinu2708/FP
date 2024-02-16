import os

from evaluation.evaluator import aggregate_scores


def calcultate_metrics(segmentation_folder, label_folder, json_output_file=None):
    """"
    param segmentation_folder: folder with the segmentations under nifty format
    param label_folder: folder with the grond truth images under nifty format
    param output_file: path to the output csv file with average performances
    param json_output_file: path to the json output file with performances per image
    return:
    """
    pred_gt_tuples = []
    for i, p in enumerate(os.listdir(label_folder)):
        if p.endswith('nii.gz'):
            file = os.path.join(label_folder, p)
            pred_gt_tuples.append([file, os.path.join(segmentation_folder, 'im' + p[2:])])

    scores = aggregate_scores(pred_gt_tuples, labels=[1, 2],
                              json_output_file=json_output_file,
                              json_author="Felix", num_threads=8)


if __name__ == "__main__":
    segmentation_folder = "/path/to/the/segmentation/folder/"
    label_folder = "/path/to/the/label/folder/"
    output_file = "/path/to/the/output/file.json"
    calcultate_metrics(segmentation_folder, label_folder, None, json_output_file=output_file)
