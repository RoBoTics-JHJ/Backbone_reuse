import os
import logging
from tqdm import tqdm
import numpy as np
from PIL import Image
from xml.etree import ElementTree
import matplotlib.pyplot as plt


def get_size_dict(dir_path):
    """
    get min from (width, height) of an image >> if width is min >> size_dict = {width:+1, ...}

    :return: {min_size of an image:the number of images, ...}
            ex) {92:3, 104:5} means there are 3 pics of 92 size and 5 pics of 104 size.
    """
    img_list = os.listdir(dir_path)

    size = {}  # {min size of an image [int] : the number of images [int], ... }
    for img in img_list:
        img = Image.open(os.path.join(dir_path, img))
        img_size = img.size
        if size.get(min(img_size)) is None:
            size[min(img_size)] = 0
        size[min(img_size)] += 1

    return size


def sum_num_of_imgs(dir_path, count_start=0, count_end=0):
    """
    :return: total number of images, range from count_start to count_end
    """
    size_dict = get_size_dict(dir_path)
    size_sort = sorted(size_dict.keys())
    print(f"The number of image sizes is '{len(size_sort)}'")
    if count_end == 0:
        count_end = len(size_sort) - 1
    selected_size = size_sort[count_start:count_end + 1]
    print(f'You choose a range from "{count_start} index to {count_end} index"\n'
          f'It means size "{size_sort[count_start]} to {size_sort[count_end]}"')

    total_imgs = 0
    for size in selected_size:
        total_imgs += size_dict[size]

    print(f'\nThe number of total images: {sum(size_dict.values())}')
    print(f'The number of chosen images: {total_imgs}')


def get_size_plot(dir_path, title='title', count_start=0, count_end=0, save=False):
    size_dict = get_size_dict(dir_path)

    size_sort = sorted(size_dict.keys())
    num_imgs = []  # the number of images for each size

    for min_size in size_sort:
        num_imgs.append(size_dict[min_size])

    plt.title(title)
    plt.xlabel("Image size")
    plt.ylabel("The number of images")
    plt.axis([size_sort[0], size_sort[-1], 0, max(num_imgs)])
    plt.bar(size_sort, num_imgs)
    if save is False:
        plt.show()
    else:
        plt.savefig(f'{title} imgs plot.png')


def remove_img(dir_path, remove_start=0, remove_end=0):
    size_dict = get_size_dict(dir_path)
    size_sort = sorted(size_dict.keys())

    if remove_end == 0:
        logging.warning("You need to give 'remove_start' value")
        return

    selected_size = size_sort[remove_start:remove_end + 1]
    img_list = os.listdir(dir_path)
    imgs = [os.path.join(dir_path, img) for img in img_list]

    total = 0
    for size in selected_size:
        total += size_dict[size]

    remove_pbar = tqdm(imgs, total=total, unit='img')
    for img_path in remove_pbar:
        img = Image.open(img_path)
        img_size = min(img.size)
        for size in selected_size:
            if size == img_size:
                os.remove(img_path)
                remove_pbar.set_description(f'Removing: {img_path}')


def count_img(xml_dir, class_file):
    with open(class_file, 'r') as class_data:
        class_list = [c.rstrip('\n') for c in class_data]
    xml_file_list = os.listdir(xml_dir)
    count_dict = {}
    for c in class_list:
        count_dict[c] = 0
    for xml_file in xml_file_list:
        tree = ElementTree.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()
        for o in root.findall('object'):
            name = o.find('name').text
            count_dict[name] += 1
    print(count_dict)


# ---------------------------------------------------------------------------------------------------------------------
# img_dir = '/home/jhj/Desktop/JHJ/projects/data/paprika_6classes/for_imagefolder/4'
#
# print(len(os.listdir(img_dir)))
# sum_num_of_imgs(img_dir)
# remove_img(img_dir, 0, 159)
# get_size_plot(img_dir, 'Gray mold', save=True)
# ---------------------------------------------------------------------------------------------------------------------
# classes = ['blossom_end_rot', 'graymold', 'powdery_mildew', 'spider_mite', 'spotting_disease', 'snails_and_slugs']
classes = ['blossom_end_rot', 'graymold', 'powdery_mildew', 'spider_mite', 'spotting_disease']
plt.title('The number of images for each class')
plt.bar(0, 769, label='0: blossom_end_rot', color="cornflowerblue")
plt.bar(1, 636, label='1: graymold', color="cornflowerblue")
plt.bar(2, 1506, label='2: powdery_mildew', color="cornflowerblue")
plt.bar(3, 1075, label='3: spider_mite', color="cornflowerblue")
plt.bar(4, 2157, label='4: spotting_disease', color="orange")
# plt.bar(5, 294, label='5: snails_and_slugs', color="cornflowerblue")
plt.xlabel('Class number')
plt.ylabel('The umber of images')
plt.legend()
# plt.show()
plt.savefig('5classes_cropped.png')
