import argparse
import csv
import tensorflow as tf

from models.research.object_detection.utils import dataset_util

# TODO add param to define path to training imgs and test images
# Non sicura cosa sia encoded_img_data, gli esempi sono contrastanti
def create_tf_wally(encoded_img_data):
  # TODO(user): Populate the following variables from your example.
  height = None # Image height
  width = None # Image width
  filename = None # Filename of the image. Empty if image is not from file
  # encoded_image_data = None # Encoded image bytes 
  image_format = None # b'jpeg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = ['wally'] # List of string class name of bounding box (1 per box)
  classes = [1] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(args):
  writer = tf.io.TFRecordWriter(args.output_path)

  # TODO Extract info from annotations.csv e metterle in un dict

  annotations = {} # Dict filename : annotations
  with open('./data/annotations.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print('Row values: {}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
  #images = 

  '''
  for img in images:
    tf_img = create_tf_wally(img)
    writer.write(tf_img.SerializeToString())

  writer.close()
  '''

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Create TF records binary files for train and test.')
  parser.add_argument('--input-path', type=str, help='Path to input folder that contains the dataset of images (train or test)')
  parser.add_argument('--output-path', type=str, help='Path to output TFRecord')

  args = parser.parse_args()
  main(args) # ????
  tf.compat.v1.app.run()