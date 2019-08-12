from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from loguru import logger

LBL_SHAPE = [256, 1600]


def create_train_img_generator(train_image_files):
    logger.info('creating train img generator')
    return map(
        lambda train_image_file: np.array(Image.open(train_image_file)),
        train_image_files
    )


def create_annotation_img(annotations):
    logger.info('creating train lbl generator')
    lbl = np.zeros(LBL_SHAPE)
    lbl = lbl.reshape(np.prod(lbl.shape), order='F')
    for id_, annotation in enumerate(annotations):
        if annotation[1] != -1:
            image_id = id_ + 1
            pixels = annotation[1].split()
            index = 0
            while index < len(pixels):
                start_index = int(pixels[index])
                end_index = int(pixels[index]) + int(pixels[index + 1])
                lbl[start_index:end_index] = image_id
                index = index + 2
    lbl = lbl.reshape(LBL_SHAPE, order='F')
    return lbl


def main():
    root_data_dir = Path(
        '~/git/severstal_steel/data/severstal-steel-defect-detection'
    ).expanduser()
    train_image_dir = root_data_dir / 'train_images'
    train_img_files = [d_ for d_ in train_image_dir.glob('*')]
    train_annotation_file = root_data_dir / 'train.csv'
    train_annotations = pd.read_csv(train_annotation_file).fillna(-1)
    train_img_gen = create_train_img_generator(train_img_files)
    train_lbl_gen = map(
        lambda train_img_file: create_annotation_img(
            train_annotations[
                train_annotations.ImageId_ClassId.str.startswith(
                    f'{train_img_file.name}'
                )
            ].values
        ),
        train_img_files
    )

    first_img = next(train_img_gen)
    first_lbl = next(train_lbl_gen)

    _, axs = plt.subplots(2, 1)
    axs[0].imshow(first_img)
    axs[1].imshow(first_lbl)
    plt.show()


if __name__ == '__main__':
    main()
