import os
from PIL import Image


def resize_images(original_image_path, out_image_path, resized_size):
    '''
    This function is used to resize our images and make all the images are with uniform size
    :param original_image_path: input images directory
    :param out_image_path: output images directory
    :param resized_size: resized size
    :return:
    '''
    # make sure that the output image path exists.
    if not os.path.exists(out_image_path):
        os.makedirs(out_image_path)
    original_images = os.listdir(original_image_path)
    num_images = len(original_images)
    for i, image in enumerate(original_images):
        with open(os.path.join(original_image_path, image), 'rb') as file:
            with Image.open(file) as img:
                img = img.resize(resized_size, Image.ANTIALIAS)
                img.save(os.path.join(out_image_path, image), img.format)
        if i % 200 == 0:
            print("[{}/{}] Resize completed".format(i, num_images))


if __name__ == '__main__':
    resize_images('./data/train2014/', './data/resized2014/', [256,256])
