from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import tensorflow as tf
import matplotlib
from PIL import Image
import matplotlib.patches as patches
import argparse

sys.path.append("./models/research/")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from utils.inference_utils import create_tiles, recombine_tiles

'''
Take an original scene from Wally's books, split the images into
tiles [256, 256], perform object detection on each tiles separately and 
return the original image with the tiles recombined and a mask 
drawn on Wally (if found).
'''

def main(args):

    def draw_box(box, image_np):
        # Expand the box by 50%
        box += np.array([-(box[2] - box[0])/2, -(box[3] - box[1])/2, (box[2] - box[0])/2, (box[3] - box[1])/2]) 

        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)

        # Draw blurred boxes around box
        ax.add_patch(patches.Rectangle(
            (0,0),
            box[1]*image_np.shape[1], 
            image_np.shape[0],
            linewidth=0,
            edgecolor='none',
            facecolor='w',
            alpha=0.8))

        ax.add_patch(patches.Rectangle((box[3]*image_np.shape[1],0),image_np.shape[1], image_np.shape[0],linewidth=0,edgecolor='none',facecolor='w',alpha=0.8))
        ax.add_patch(patches.Rectangle((box[1]*image_np.shape[1],0),(box[3]-box[1])*image_np.shape[1], box[0]*image_np.shape[0],linewidth=0,edgecolor='none',facecolor='w',alpha=0.8))
        ax.add_patch(patches.Rectangle((box[1]*image_np.shape[1],box[2]*image_np.shape[0]),(box[3]-box[1])*image_np.shape[1], image_np.shape[0],linewidth=0,edgecolor='none',facecolor='w',alpha=0.8))

        return fig, ax
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(args.model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    def load_image_into_numpy_array(image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

    label_map = label_map_util.load_labelmap(args.label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            img_path = os.path.join(args.image_dir, args.filename)
            #print('img path', img_path)
            # Generate tiles
            create_tiles(img_path, (256, 256), (256, 256))
            tmp_dir = os.path.join(args.image_dir, "tmp")
            # Perform object detection on each tile
            for tile in os.listdir(tmp_dir):

                tile_path = os.path.join(tmp_dir, tile)
                image_np = load_image_into_numpy_array(Image.open(tile_path))
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

                if scores[0][0] < 0.7: # Try with different thresholds
                    print('Wally not found :(')
                    continue # Go to next loop iteration

                else:
                    print('Wally found')
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=6)

                    im = Image.fromarray(image_np)
                    im.save(tile_path) # Overwrite image with bounding box


            # Recombine tiles into bigger image but now with bounding boxes
            recombine_tiles(tmp_dir, img_path, tile_size=(256, 256), offset=(256, 256))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--label_map', help='Path to label map for classes')
    parser.add_argument('--model_path', help='Path to frozen inference graph') # TODO Could pass only path to image and retrieve dir from that
    parser.add_argument('--image_dir', help='Path to folder containing the image used for inference/object detection')
    parser.add_argument('--filename', help='Filename of the image chosen for object detection')
    args = parser.parse_args()
    main(args)