# WALL-Y
Finding Wally using neural networks

## Getting started

### Prerequisites
* Python 3.7
* pip >= 19

### Installation
First clone the project repository to get the source
```bash
git clone https://github.com/giulic3/WALL-Y.git
```

Run the following to initialize a copy of the Tensorflow Object Detection API project

```bash
git submodule init
git submodule update
```

(Carefully) follow the instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) to install the API.

Create a virtualenv and install the required packages to run WALL-Y

```bash
virtualenv -p /path/to/python3.7 venv
source venv/bin/activate
pip install -r requirements.txt

```

## Project structure
* data/ : contains the original dataset, the dataset with cropped images and the annotations (xml and csv)
* utils/ : contains the script used to reduce the original images in size (crop_images.py) and other helper functions to be used during inference
* . /contains Python scripts for training (model_main.py) and performing object detection (find_wally_.py, inference.py)

## How to run
...
