# PyQt5 GUI for creating TFRecord Files easily!
This GUI for Tensorflows TFRWriter more comfortable with no need to code each time or pass long paths as commandline arguments.
It is fully compatible with Tensorflow >= 2.0 and allows to write all annotations to a .csv file for further use (e.g. YOLO).

![TFRWriter GUI](/screenshot.png)

Runs fine on macOS and Windows (...and probably Linux).

## What are TFRecord files?
A TFRecord file is a binary format to store Tensorflow datasets in an optimized way to achieve higher performance for read operations during training (https://www.tensorflow.org/guide/data_performance). It is widely used for object detection datasets but in general, lots of other datatypes can be saved this way.

## Requirements
- Tensorflow >= 2.0
- Tensorflow Object Detection API
- numpy
- pandas
- pillow
- absl-py
- pyqt5

## Usage
Execute the script:

`python3 tfrecord_generate_gui.py`

That's all!
The GUI will appear shortly and you can point to the paths of Images, XMLs, .pbtxt class label map und the output path of the .record file which will be created after clicking the start button.