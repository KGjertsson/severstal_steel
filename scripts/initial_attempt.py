from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from kss import generators

LBL_SHAPE = [256, 1600]


def main():
    # define variables
    root_data_dir = Path(
        '~/git/severstal_steel/data/severstal-steel-defect-detection'
    ).expanduser()
    train_image_dir = root_data_dir / 'train_images'
    train_img_files = [d_ for d_ in train_image_dir.glob('*')]
    train_annotation_file = root_data_dir / 'train.csv'
    train_annotations = pd.read_csv(train_annotation_file).fillna(-1)
    validation_split = 0.7
    # TODO: include batch size in generators
    batch_size = 32
    # initialize generators
    train_gen = generators.create_train_gens(
        train_img_files=train_img_files[:int(len(train_img_files) * validation_split)],
        train_annotations=train_annotations[:int(len(train_annotations) * validation_split)],
        lbl_shape=LBL_SHAPE,
        shuffle=True
    )
    val_gen = generators.create_train_gens(
        train_img_files=train_img_files[int(len(train_img_files) * validation_split):],
        train_annotations=train_annotations[int(len(train_annotations) * validation_split):],
        lbl_shape=LBL_SHAPE,
        shuffle=True
    )
    # create CNN
    input_ = tf.keras.layers.Input(shape=LBL_SHAPE)
    c1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(input_)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)
    c2 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(c2)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)
    c3 = tf.keras.layers.UpSampling2D()(c2)
    c3 = tf.keras.layers.add([c3, c1])
    c3 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(c3)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)
    output = tf.keras.layers.Conv2D(5, (1, 1), padding='same')(c3)
    output = tf.keras.layers.Activation('softmax')(output)
    model = tf.keras.models.Model(input=input_, output=output)
    model.compile('adam', loss='categorical_crossentropy', metrics=[])
    # train CNN
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=None,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_data=val_gen,
        validation_steps=None,
        workers=1,
        use_multiprocessing=True,
        shuffle=True
    ).history
    # TODO: set the correct arguments, specifically the steps per epoch and shuffle
    logger.info(f'training history: {history}')
    first_img, first_lbl = next(train_gen)
    _, axs = plt.subplots(2, 1)
    axs[0].imshow(first_img)
    axs[1].imshow(first_lbl)
    plt.show()


if __name__ == '__main__':
    main()
