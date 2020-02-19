import cv2
import math
from PIL import Image
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
            #print('filename', filename)

            dir_path = os.path.join(os.path.dirname(img_path), "tmp")#_" + filename)
            #print('dir_path', dir_path)
            if not os.path.exists(dir_path):
               os.mkdir(dir_path)
            cv2.imwrite(os.path.join(dir_path, filename + str(i) + "_" + str(j) + ".jpg"), tile_img)

# img_dir sarebbe la tmp/
def recombine_tiles(img_dir, original_image_path, tile_size, offset):

    original_image = cv2.imread(original_image_path)
    img_shape = original_image.shape

    # Concat a list of images horizontally
    def get_concat_h_multi(im_list):
        width = im_list[0].width
        total_width = width * len(im_list)
        height = im_list[0].height
        dst = Image.new('RGB', (total_width, height))
        pos_x = 0
        for im in im_list:
            dst.paste(im, (pos_x, 0))
            pos_x += im.width
        return dst

    # Concat a list of images vertically
    def get_concat_v_multi(im_list):
        width = im_list[0].width
        height = im_list[0].height
        total_height = height * len(im_list)
        dst = Image.new('RGB', (width, total_height))
        pos_y = 0
        for im in im_list:
            dst.paste(im, (0, pos_y))
            pos_y += im.height
        return dst

    def get_concat_tiles(im_list_2d):
        im_list_v = [get_concat_h_multi(im_list_h) for im_list_h in im_list_2d]
        return get_concat_v_multi(im_list_v)

    max_i = int(math.ceil(img_shape[0]/(offset[1] * 1.0)))
    max_j = int(math.ceil(img_shape[1]/(offset[0] * 1.0)))

    
    im_list_2d = []
    row_list = []
    j = 0
    for tile_image in sorted(os.listdir(img_dir)):
        print(tile_image)
        if j < max_j:
            row_list.append(Image.open(os.path.join(img_dir, tile_image)))
            j += 1

        else: # Switch to new row
            im_list_2d.append(row_list) # Append old row and reset
            print(len(row_list))
            j = 1
            row_list = [Image.open(os.path.join(img_dir, tile_image))]
    
    print('shape', len(im_list_2d))
    get_concat_tiles(im_list_2d).save(os.path.join(img_dir, 'result.jpg'))


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp_dir') # path to tmp dir with tiles
    parser.add_argument('--original_image_path') # path to whole image
    args = parser.parse_args()
    # Recombine tiles into bigger image but now with bounding boxes
    recombine_tiles(args.tmp_dir,
                    args.original_image_path,
                    tile_size=(256, 256), 
                    offset=(256, 256))