import copy
import random

from PIL import Image
from loguru import logger
import numpy as np


def create_train_gens(train_img_files, train_annotations, lbl_shape, shuffle=True):
    logger.info('creating generators...')
    files = copy.deepcopy(train_img_files)
    return _train_img_lbl_gen(files, shuffle, train_annotations, lbl_shape)


def _train_img_lbl_gen(files, shuffle, train_annotations, lbl_shape):
    if shuffle:
        files = random.shuffle(files)
    img_gen, lbl_gen = _init_generators(files, train_annotations, lbl_shape)
    while True:
        try:
            img, lbl = next(img_gen), next(lbl_gen)
            yield img, lbl
        except StopIteration:
            if shuffle:
                files = random.shuffle(files)
            img_gen, lbl_gen = _init_generators()
            img, lbl = next(img_gen), next(lbl_gen)
            yield img, lbl


def _init_generators(files, train_annotations, lbl_shape):
    img_gen = map(lambda img_file: np.array(Image.open(img_file)), files)
    lbl_gen = map(
        lambda lbl_file: _create_annotation_img(
            train_annotations[
                train_annotations.ImageId_ClassId.str.startswith(
                    f'{lbl_file.name}'
                )
            ].values,
            lbl_shape),
        files
    )
    return img_gen, lbl_gen


def _create_annotation_img(annotations, lbl_shape):
    lbl = np.zeros(lbl_shape)
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
    lbl = lbl.reshape(lbl_shape, order='F')
    return lbl
