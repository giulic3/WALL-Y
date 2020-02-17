import os
from impy.ObjectDetectionDataset import ObjectDetectionDataset

def main():
	# Define the path to images and annotations
	images_path:str = os.path.join(os.getcwd(), "attempt2", "dataset")
	annotations_path:str = os.path.join(os.getcwd(), "attempt2", "annotations", "xml")
	# Define the name of the dataset
	dbName:str = "wally"

	print(images_path)
	print(annotations_path)
	# Create an object of ObjectDetectionDataset
	obda:any = ObjectDetectionDataset(imagesDirectory=images_path, annotationsDirectory=annotations_path, databaseName=dbName)
	# Reduce the dataset to smaller crops of shape 256x256.
	offset:list=[256, 256]
	images_output_path:str = os.path.join(os.getcwd(), "attempt2", "dataset_cropped")
	annotations_output_path:str = os.path.join(os.getcwd(), "attempt2", "annotations", "xml_cropped")

	print(images_output_path)
	print(annotations_output_path)

	obda.reduceDatasetByRois(offset = offset, outputImageDirectory = images_output_path, outputAnnotationDirectory = annotations_output_path)

if __name__ == "__main__":
	main()