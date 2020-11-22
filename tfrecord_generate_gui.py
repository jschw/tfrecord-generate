#!/usr/bin/python

"""
TFRecord Generator GUI

This simple user interface is based on Qt5
and helps creating Tensorflow record files (for object detection)
from Pascal VOC annotations.
All annotations can optionally be written to a .csv file.

"""

import sys
from PyQt5.QtWidgets import (
    QWidget, QToolTip,
    QPushButton, QLabel,
    QApplication, QGridLayout,
    QSpacerItem, QSizePolicy,
    QLineEdit, QFileDialog,
    QCheckBox)
from PyQt5.QtGui import QFont
from PyQt5 import QtCore


import os
import glob
import pandas as pd
import io
import shutil
from PIL import Image
import ntpath
import pathlib
import numpy as np
from collections import namedtuple
import xml.etree.ElementTree as ET

# Turn off TF warning messsages -> Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util


from absl import logging
# Setting loglevel for custom logger
logging.set_stderrthreshold('info')
logging.set_verbosity('info')




class TfrecordGenerate(QWidget):

    InputImagePath = ''
    InputXmlPath = ''
    InputPbtxtPath = ''
    OutputPath = ''
    SaveCsvPath = ''
    SaveCsv = False

    # Conversion functions
    def xml_to_csv(self, path):

        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
        column_name = ['filename', 'width', 'height',
                       'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df


    def class_text_to_int(self, label_map_dict, row_label):
        return label_map_dict[row_label]


    def split(self, df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


    def create_tf_example(self,label_map_dict, group, path):
        with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(self.class_text_to_int(label_map_dict, row['class']))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example


    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        QToolTip.setFont(QFont('SansSerif', 10))

        grid = QGridLayout()
        self.setLayout(grid)
        grid.setSpacing(5)

        lbl10 = QLabel('Generate Tensorflow .record File<br>', self)
        lbl10.setFont(QFont('SansSerif', 14))
        grid.addWidget(lbl10,1,0)

        lbl13 = QLabel('<b>Input Image Directory:</b>', self)
        grid.addWidget(lbl13,2,0)
        self.editInputImgPath = QLineEdit()
        grid.addWidget(self.editInputImgPath,3,0)
        btnImgPath = QPushButton('Browse', self)
        btnImgPath.setObjectName('btnImg')
        grid.addWidget(btnImgPath,3,1)

        lbl14 = QLabel('<b>Input XML Directory:</b>', self)
        grid.addWidget(lbl14,4,0)
        self.editInputXmlPath = QLineEdit()
        grid.addWidget(self.editInputXmlPath,5,0)
        btnXmlPath = QPushButton('Browse', self)
        btnXmlPath.setObjectName('btnXml')
        grid.addWidget(btnXmlPath,5,1)

        lbl15 = QLabel('<b>Class Label File (.pbtxt):</b>', self)
        grid.addWidget(lbl15,6,0)
        self.editClassLabelFile = QLineEdit()
        grid.addWidget(self.editClassLabelFile,7,0)
        btnPbtxtPath = QPushButton('Browse', self)
        btnPbtxtPath.setObjectName('btnPbtxt')
        grid.addWidget(btnPbtxtPath,7,1)

        lbl16 = QLabel('<b>Output Directory:</b>', self)
        grid.addWidget(lbl16,8,0)
        self.editOutputPath = QLineEdit()
        grid.addWidget(self.editOutputPath,9,0)
        btnOutPath = QPushButton('Browse', self)
        btnOutPath.setObjectName('btnOut')
        grid.addWidget(btnOutPath,9,1)

        self.checkBoxCSV = QCheckBox('Write annotations to .csv file')
        self.checkBoxCSV.stateChanged.connect(self.checkBoxChangedAction)
        grid.addWidget(self.checkBoxCSV,10,0)

        lbl17 = QLabel(' ', self)
        grid.addWidget(lbl17,11,0)

        btnStart = QPushButton('Generate + Write TFRecord File', self)
        btnStart.setToolTip('Write a TFRecord File to disk according to the given parameters.')
        btnStart.resize(btnStart.sizeHint())
        btnStart.setFixedHeight(50)
        grid.addWidget(btnStart,12,0)

        qbtn = QPushButton('Quit', self)
        qbtn.resize(qbtn.sizeHint())
        grid.addWidget(qbtn,13,0)

        # Connect events
        qbtn.clicked.connect(QApplication.instance().quit)
        btnStart.clicked.connect(self.buttonStartClicked)

        btnImgPath.clicked.connect(self.buttonBrowseDir)
        btnXmlPath.clicked.connect(self.buttonBrowseDir)
        btnOutPath.clicked.connect(self.buttonSaveFile)
        btnPbtxtPath.clicked.connect(self.buttonBrowseFile)

        self.setGeometry(400, 300, 600, 300)
        self.setWindowTitle('Generate Tensorflow .record File')
        self.show()

    
    def checkBoxChangedAction(self, state):
        if (QtCore.Qt.Checked == state):
            self.SaveCsv = True
        else:
            self.SaveCsv = False


    def buttonStartClicked(self):
        sender = self.sender()
        problems = False

        # Copy values to variables
        self.InputImagePath = self.editInputImgPath.text()
        self.InputXmlPath = self.editInputXmlPath.text()
        self.InputPbtxtPath = self.editClassLabelFile.text()
        self.OutputPath = self.editOutputPath.text()

        if self.InputImagePath == '':
            logging.warning('==> No Image Directory selected.')
            problems = True

        if self.InputXmlPath == '':
            logging.warning('==> No XML Directory selected.')
            problems = True

        if self.InputPbtxtPath == '':
            logging.warning('==> No .pbtxt class label File selected.')
            problems = True

        if self.OutputPath == '':
            logging.warning('==> No Output Path selected.')
            problems = True

        if problems: return

        if self.SaveCsv:
            self.SaveCsvPath = os.path.dirname(self.OutputPath) +  '/annotations.csv'
        else: self.SaveCsvPath = ''

        logging.info('==> Start generating the TFRecord file...')

        label_map = label_map_util.load_labelmap(self.InputPbtxtPath)
        label_map_dict = label_map_util.get_label_map_dict(label_map)

        writer = tf.io.TFRecordWriter(self.OutputPath)
        examples = self.xml_to_csv(self.InputXmlPath)
        grouped = self.split(examples, 'filename')

        for group in grouped:
            tf_example = self.create_tf_example(label_map_dict, group, self.InputImagePath)
            writer.write(tf_example.SerializeToString())
        writer.close()

        logging.info('==> Successfully created the TFRecord file: {}'.format(self.OutputPath))

        if self.SaveCsv:
            examples.to_csv(self.SaveCsvPath, index=None)
            logging.info('==> Successfully created the CSV file: {}'.format(self.SaveCsvPath))


 

    def buttonBrowseDir(self):
        sender = self.sender()
        btn_objname = str(sender.objectName())

        srcdirname, filename = os.path.split(os.path.abspath(__file__))

        #home_dir = str(Path.home())
        dirname = QFileDialog.getExistingDirectory(self, 'Open Folder', srcdirname)

        if not dirname: return

        if btn_objname == 'btnImg':
            self.imgFilePath = dirname
            self.editInputImgPath.setText(dirname)
        elif btn_objname == 'btnXml':
            self.xmlFilePath = dirname
            self.editInputXmlPath.setText(dirname)


    def buttonBrowseFile(self):
        sender = self.sender()
        btn_objname = str(sender.objectName())

        dirname, filename = os.path.split(os.path.abspath(__file__))

        #home_dir = str(Path.home())
        fname = QFileDialog.getOpenFileName(self, 'Open File', dirname)

        if not fname: return

        if btn_objname == 'btnPbtxt':
            self.editClassLabelFile.setText(fname[0])
            self.editInputPbtxtPath = fname[0]


    def buttonSaveFile(self):
        sender = self.sender()
        btn_objname = str(sender.objectName())

        dirname, filename = os.path.split(os.path.abspath(__file__))

        #home_dir = str(Path.home())
        fname = QFileDialog.getSaveFileName(self, 'Save File', dirname)

        if not fname: return

        if btn_objname == 'btnOut':
            self.imgFilePath = dirname
            self.editOutputPath.setText(fname[0])
        


def main():

    app = QApplication(sys.argv)
    mainWindow = TfrecordGenerate()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()