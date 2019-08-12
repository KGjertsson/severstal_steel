from pathlib import Path


def load_train_images(root_data_dir):
    train_image_dir = root_data_dir / 'train_images'

    for train_image_file in train_image_dir.glob('*'):
        print(train_image_file)


def convert_labels_to_images():
    pass


def main():
    root_data_dir = Path(
        '~/git/severstal_steel/data/severstal-steel-defect-detection'
    ).expanduser()
    load_train_images(root_data_dir)
    convert_labels_to_images()


if __name__ == '__main__':
    main()
