# comment out below line to enable tensorflow outputs
# import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from absl import app, flags, logging
# from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import easyocr
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()

# flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
# flags.DEFINE_string('weights', './checkpoints/yolov4-416',
#                     'path to weights file')
# flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
# flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
# flags.DEFINE_list('images', './data/images/car1.jpg', 'path to input image')
# flags.DEFINE_string('output', './detections/', 'path to output folder')
# flags.DEFINE_float('iou', 0.45, 'iou threshold')
# flags.DEFINE_float('score', 0.50, 'score threshold')
# flags.DEFINE_boolean('count', False, 'count objects within images')
# flags.DEFINE_boolean('dont_show', False, 'dont show image output')
# flags.DEFINE_boolean('info', False, 'print info on detections')
# flags.DEFINE_boolean('crop', False, 'crop detections from images')
# flags.DEFINE_boolean('ocr', False, 'perform generic OCR on detection regions')
# flags.DEFINE_boolean('plate', True, 'perform license plate recognition')

class ANPRYOLOV4:
    def yolov4model(self):
        print("yolov4_ANPR model loading \n")
        saved_model_loaded = tf.saved_model.load('E:\python projects\Vehicle-Detection-Classification-YOLO-MobileNet-main\checkpoints\yolov4-416', tags=[tag_constants.SERVING])
        # infer = saved_model_loaded.signatures['serving_default']
        print("\t yolov4_ANPR model loaded")
        return saved_model_loaded
    def notmain(self, saved_model_loaded, origin_image):

        # STRIDES = np.array(cfg.YOLO.STRIDES)
        # ANCHORS = get_anchors(cfg.YOLO.ANCHORS)
        # NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))
        # XYSCALE = XYSCALE = cfg.YOLO.XYSCALE


        # cv2.imwrite("G:\python projects\Vehicle-Detection-Classification-YOLO-MobileNet-main\Image_data\capture.png", origin_image)
        # origin_image_path = 'E:\python projects\Vehicle-Detection-Classification-YOLO-MobileNet-main\data\images\Capture3.png'
        # origin_image_path = "E:\python projects\Vehicle-Detection-Classification-YOLO-MobileNet-main\Image_data\cbc.PNG" # saved cropped picture
        # origin_image = cv2.imread(origin_image_path)
        # os.remove(origin_image_path)
        input_size = 416
        images = np.uint8(origin_image)
        # images = np.asarray(images)

        original_image = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        img_plot = plt.imshow(image_data)
        plt.show()
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
        print(images_data.shape)
        # saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # run non max suppression on detections
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.50
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = original_image.shape
        print(boxes[0, 1])
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        # hold all detection data in one variable
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        image = utils.draw_bbox(original_image, pred_bbox, False, allowed_classes=allowed_classes, read_plate=True)

        # image = np.uint8(image)
        image = Image.fromarray(image.astype(np.uint8))

        out_box, a, b, c = pred_bbox
        coor = out_box[0]
        # c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cropped_image = image.crop((coor[0], coor[1], coor[2], coor[3]))
        cropped_image = np.uint8(cropped_image.resize((360, 180)))
        plt_crop_image = plt.imshow(cropped_image)
        plt.show()


        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        print(result)
        return result

# model = ANPRYOLOV4().yolov4model()
# ANPRYOLOV4().notmain(model)