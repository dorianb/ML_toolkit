import os


def get_image_label(folder_path):
    """

    Args:
        folder_path: the path to the Caltech dataset folder

    Returns:
        list of tuple with image path, image filename, label id and label name
    """
    result = []

    for label in os.listdir(folder_path):

        label_id, label_name = label.split(".")
        label_path = os.path.join(folder_path, label)

        for image_filename in os.listdir(label_path):

            image_path = os.path.join(label_path, image_filename)
            result.append((image_path, image_filename, label_id, label_name))

    return result
