# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import cv2
import os
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
# from keras import backend as K
import tensorflow.compat.v1.keras.backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

tf.compat.v1.disable_eager_execution()


# I changed nms score from 0.3 to 0.45
# Reference : https://medium.com/analytics-vidhya/you-only-look-once-yolo-implementing-yolo-in-less-than-30-lines-of-python-code-97fb9835bfd2#:~:text=YOLO%20uses%20Non%2DMaximal%20Suppression,this%20NMS%20threshold%20to%200.6.
# Non-Maximal Suppression(NMS)Score and Intersection Over Union Threshold(IOU) can improve our output F1 scores
class YOLOV4(object):
    # If using Tiny Yolo use this (Tiny YOLO is a much smaller and faster network which can process 220	fps)
    # _defaults = {
    #     "model_path": 'model_data/yolo-tiny.h5',
    #     "anchors_path": 'model_data/yolo_tiny_anchors.txt',
    #     "classes_path": 'model_data/coco_classes.txt',
    #     "score" : 0.2,
    #     "iou" : 0.2,
    #     "model_image_size" : (416, 416),
    #     "gpu_num" : 1,
    # }
    # To use YOLO model uncomment bellow line, it will be slower but with a better mAP then TinyYolo
    # you can vary score and iou as the detection depends on these factors

    # If using Yolo use this
    _defaults = {
        "model_path": 'model_data/yolo_v4_ANPR.h5',
        "anchors_path": 'model_data/yolov4_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.50,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = tf.compat.v1.keras.backend.get_session()  # K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):

        start = timer()
        printimage = plt.imshow(image)
        plt.show()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 800

        position_list2 = list()
        # my_score = list()
        # my_class = list()
        # my_boxes = list()
        # for index in range(len(out_classes)):
        #     if out_classes[index] == (2 or 5 or 7):
        #         my_class.append(out_classes[index])
        #         my_score.append(out_scores[index])
        #         my_boxes.append(out_boxes[index])

        for i, c in list(enumerate(out_classes)):  # the list was reversed but i removed it
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            # label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.1).astype('int32'))
            left = max(0, np.floor(left + 0.1).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.1).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.1).astype('int32'))
            # print(label, (left, top), (right, bottom))

            position = dict()
            position["class"] = predicted_class
            position["left"] = left
            position["top"] = top
            position["right"] = right
            position["bottom"] = bottom

            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            # else:
            #     text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=self.colors[c])
            # draw.text(text_origin, label, fill=(0, 0, 0), font=font)

            imgplot = plt.imshow(image)
            plt.show()

            del draw
            position_list2.append(position)

        end = timer()
        print(end - start)
        return image, position_list2

    def close_session(self):
        self.sess.close()

yolov4 = YOLOV4()
image = './data/images/car1.jpg'
origin_image = cv2.imread(image)
origin_image = np.uint8(origin_image)
image2, position_list2 = yolov4.detect_image(origin_image)