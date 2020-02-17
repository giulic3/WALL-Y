import cv2
import math
import os
import argparse

def create_tiles(img_path, tile_size, offset):
    """
    :param img_path: path to img to be splitted into tiles
    :param tile_size: tuple of (int, int) containing dimension of tiles, def 256
    :param offset
    """

    img = cv2.imread(img_path)

    img_shape = img.shape
    # tile_size = (256, 256)
    # offset = (256, 256)

    for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
            tile_img = img[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
            
            # Write tiles to tmp directory

            filename = img_path.split('/')[-1].split('.')[0] # Use img original name
            print('filename', filename)

            dir_path = os.path.join(os.path.dirname(img_path), "tmp")#_" + filename)
            print('dir_path', dir_path)
            if not os.path.exists(dir_path):
               os.mkdir(dir_path)
            cv2.imwrite(os.path.join(dir_path, filename + str(i) + "_" + str(j) + ".jpg"), tile_img)


#def recombine_tiles(img_dir):
