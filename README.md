# WALL-Y
Finding Wally using Object Detection and Neural Networks

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
* `data/`: contains the original dataset, the dataset with cropped images and the annotations (xml and csv)
* `utils/`: contains the script used to reduce the original images in size (crop_images.py) and other helper functions to be used during inference
* `networks/`: contains model configuration and checkpoint
* `src/`: contains Python scripts for training (`model_main.py`), performing object detection (`find_wally_.py`, `inference.py`) and graph freeze for inference

## How to run
Every command supposes the current working directory is the project root and the `venv` is activated.
#### Training/Fine-Tuning and Evaluation
```
python src/model_main.py --logtostderr --model_dir="OUTPUT_DIR" --pipeline_config_path=networks/faster_rcnn_resnet101_coco.config --num_train_steps=3000
```
Replace `"OUTPUT_DIR"` with the directory where trained model and checkpoints will be saved; `num_train_steps` states total number of training steps. Be careful when increasing this number as overfitting problems may arise.

#### Evaluation Only
```
python src/model_main.py --run_once --pipeline_config_path=networks/faster_rcnn_resnet101_coco.config --checkpoint_dir="TRAINED_MODEL_DIR" --model_dir="OUTPUT_DIR"
```
Replace `"OUTPUT_DIR"` with the path you want to save Tensorboard data to and also replace `"TRAINED_MODEL_DIR"` pointing to the path containing the trained model.  
**NB:** `"TRAINED_MODEL_DIR"` MUST point to the directory containing checkpoints and the last one will be automatically loaded; don't put names or suffixes (such as `model.ckpt`).

#### Inference Graph Export
```
python src/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=networks/faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix="TRAINED_MODEL_PREFIX" --output_directory="OUTPUT_DIR"
```
In order to use the trained model for inference, it has to be frozen with this command.  
Replace `"TRAINED_MODEL_PREFIX"` with the path of the trained model, including model name prefix (such as `model.ckpt-1000`); replace also `"OUTPUT_DIR"` to state output directory for the frozen graph.

#### Inference
```
python src/inference.py --label_map=data/mscoco_label_map.pbtxt --model_path="TRAINED_MODEL_DIR"/frozen_inference_graph.pb --image_dir=data/dataset_cropped/inference/ --filename=lake.jpg
```
Replace `"TRAINED_MODEL_DIR"` pointing to the path containing the trained model. The resulting image will be saved in `image_dir/tmp/`.

## Credits

- [Anna Avena](https://github.com/annaavena)
- [Giulia Cantini]()
- [Matteo Del Vecchio](https://github.com/matteodelv)
- [Simone Preite](https://github.com/simonepreite)
