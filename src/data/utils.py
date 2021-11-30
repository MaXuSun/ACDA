import torchvision.transforms as transforms
from PIL import Image
import os
from config.config import cfg

def get_transform(train=True):
    transform_list = []

    if cfg.DATA_TRANSFORM.RESIZE_OR_CROP == 'resize_and_crop':
        osize = [cfg.DATA_TRANSFORM.LOADSIZE, cfg.DATA_TRANSFORM.LOADSIZE]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        if train:
            transform_list.append(transforms.RandomCrop(cfg.DATA_TRANSFORM.FINESIZE))
        else:
            transform_list.append(transforms.CenterCrop(cfg.DATA_TRANSFORM.FINESIZE))

    elif cfg.DATA_TRANSFORM.RESIZE_OR_CROP == 'crop':
        if train:
            transform_list.append(transforms.RandomCrop(cfg.DATA_TRANSFORM.FINESIZE))
        else:
            transform_list.append(transforms.CenterCrop(cfg.DATA_TRANSFORM.FINESIZE))

    if train and cfg.DATA_TRANSFORM.FLIP:
        transform_list.append(transforms.RandomHorizontalFlip())

    to_normalized_tensor = [transforms.ToTensor(),
                            transforms.Normalize(mean=cfg.DATA_TRANSFORM.NORMALIZE_MEAN,
                                       std=cfg.DATA_TRANSFORM.NORMALIZE_STD)]

    transform_list += to_normalized_tensor

    return transforms.Compose(transform_list)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_with_labels(dir, classnames):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    labels = []

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            dirname = os.path.split(root)[-1]
            if dirname not in classnames:
                continue
            label = classnames.index(dirname)

            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                labels.append(label)

    return images, labels
 
def make_dataset_classwise(dir, category):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            dirname = os.path.split(root)[-1]
            if dirname != category:
                continue
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images
