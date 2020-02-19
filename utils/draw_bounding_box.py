import numpy as np
import sys
import tensorflow as tf
from PIL import Image
#import scipy.misc
import argparse

sys.path.append("./models/research/")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

'''
Look for Wally in a pic and draw it on screen with green ugly bounding boxes
and an indication of accuracy in percentage %.
WITHOUT USING MATPLOTLIB

Usage 
python find_wally_pretty.py --label_map=<PATH_TO_LABEL_MAP> --model_path=<PATH_TO_FROZEN_GRAPH> --image_path=<PATH_TO_IMAGE_FOR_DETECTION>

'''

def main(args):

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

        image_np = load_image_into_numpy_array(Image.open(args.image_path))
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

        if scores[0][0] < 0.1:
            sys.exit('Wally not found :(')

        print('Wally found')
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        im = Image.fromarray(image_np)
        im.save(args.image_path) # Overwrite image with bounding box


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--label_map')
    parser.add_argument('--model_path') # Path to frozen inference graph
    parser.add_argument('--image_path')
    args = parser.parse_args()
    main(args)
