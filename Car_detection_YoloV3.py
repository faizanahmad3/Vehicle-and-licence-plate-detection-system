# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:45:22 2020

@author: hp
"""
import sys
import colorsys
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from PIL import Image, ImageFont, ImageDraw
from PIL import Image, ImageDraw
from keras import Model
from keras.regularizers import l2
from keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization
)

# from tensorflow.keras import Model
# from tensorflow.keras.layers import (
#     Add,
#     Concatenate,
#     Conv2D,
#     Input,
#     Lambda,
#     LeakyReLU,
#     UpSampling2D,
#     ZeroPadding2D,
#     BatchNormalization
# )
# from tensorflow.keras.regularizers import l2
import wget
from final_detect import ANPRYOLOV4

model = ANPRYOLOV4().yolov4model()


def draw_outputs(img, outputs, class_names):
    '''
    Helper, util, function that draws predictons on the image.

    :param img: Loaded image
    :param outputs: YoloV3 predictions
    :param class_names: list of all class names found in the dataset
    '''
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        if classes[i] == 2 and objectness[i] > 0.9:
            oimg = img[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
            # oimg = img.crop((x1y1[0], x1y1[1], x2y2[0], x2y2[1]))
            # cv2.imwrite("E:\python projects\Vehicle-Detection-Classification-YOLO-MobileNet-main\Image_data\cbc.jpg",
            #             oimg)
            # cv2.imshow("img box", oimg)

            ANPRYOLOV4().notmain(model, oimg)
        # img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        # img = cv2.putText(img, '{} {:.4f}'.format(
        #     class_names[int(classes[i])], objectness[i]),
        #                   x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img

def load_darknet_weights(model, weights_file):
    '''
    Helper function used to load darknet weights.
    
    :param model: Object of the Yolo v3 model
    :param weights_file: Path to the file with Yolo V3 weights
    '''

    # Open the weights file
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    # Define names of the Yolo layers (just for a reference)
    layers = ['yolo_darknet',
              'yolo_conv_0',
              'yolo_output_0',
              'yolo_conv_1',
              'yolo_output_1',
              'yolo_conv_2',
              'yolo_output_2']

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):

            if not layer.name.startswith('conv2d'):
                continue

            # Handles the special, custom Batch normalization layer
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416

yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def DarknetConv(x, filters, kernel_size, strides=1, batch_norm=True):
    '''
    Call this function to define a single Darknet convolutional layer
    
    :param x: inputs
    :param filters: number of filters in the convolutional layer
    :param kernel_size: Size of kernel in the Conv layer
    :param strides: Conv layer strides
    :param batch_norm: Whether or not to use the custom batch norm layer.
    '''
    # Image padding
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'

    # Defining the Conv layer
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)

    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    '''
    Call this function to define a single DarkNet Residual layer
    
    :param x: inputs
    :param filters: number of filters in each Conv layer.
    '''
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    '''
    Call this function to define a single DarkNet Block (made of multiple Residual layers)
    
    :param x: inputs
    :param filters: number of filters in each Residual layer
    :param blocks: number of Residual layers in the block
    '''
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None):
    '''
    The main function that creates the whole DarkNet.
    '''
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def YoloConv(filters, name=None):
    '''
    Call this function to define the Yolo Conv layer.
    
    :param flters: number of filters for the conv layer
    :param name: name of the layer
    '''

    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    '''
    This function defines outputs for the Yolo V3. (Creates output projections)
     
    :param filters: number of filters for the conv layer
    :param anchors: anchors
    :param classes: list of classes in a dataset
    :param name: name of the layer
    '''

    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return yolo_output


def yolo_boxes(pred, anchors, classes):
    '''
    Call this function to get bounding boxes from network predictions
    
    :param pred: Yolo predictions
    :param anchors: anchors
    :param classes: List of classes from the dataset
    '''

    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    # Extract box coortinates from prediction vectors
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    # Normalize coortinates
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
             tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=0.5,
        score_threshold=0.6
    )

    return boxes, scores, classes, valid_detections


def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80):
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


'''def weights_download(out='models/yolov3.weights'):
    _ = wget.download('https://pjreddie.com/media/files/yolov3.weights', out='models/yolov3.weights', bar=bar_progress)
'''

# weights_download()  # to download weights
yolo = YoloV3()
print("loading yolov3 weights in this file")
load_darknet_weights(yolo, 'E:\python projects\Vehicle-Detection-Classification-YOLO-MobileNet-main\yolov3.weights')
print("yolov3 weights loaded")

carVideo = cv2.VideoCapture(r"E:\python projects\Vehicle-Detection-Classification-YOLO-MobileNet-main\test\my9.mp4")

# code from video reader start

# carVideo.set(cv2.CAP_PROP_FPS,30)
# fps = carVideo.get(cv2.CAP_PROP_FPS)
# print("FPS : ",fps)
# print("<<::::: DISPLAYING ORIGINAL VIDEO :::::>>")
# Frame = []
# while True:
#     frame_no = 0
#     ret1, frame1 = carVideo.read()
#     if ret1 == False:
#         break
#     # Code reference: yolo.py => detect_video
#     image = Image.fromarray(frame1)
#     # Frame.append(frame_no,image)
#     image = np.asarray(image)
#     # https://pythonexamples.org/python-opencv-write-text-on-image-puttext/
#     position = (10,20)
#     cv2.putText(image,"Original Video",position,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2) #font stroke
#     position = (10,50)
#     cv2.putText(image,"Frame : "+str(frame_no),position,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2) #font stroke
#     result = np.asarray(image)
#     cv2.imshow("Original Video", result)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     frame_no = frame_no + 1
#     if frame_no == 100:
#         break
#
# # carVideo.release()
# # cv2.destroyAllWindows()
# print("<<::::: DISPLAYING ORIGINAL VIDEO EXIT :::::>>")
# code from video reader end

counter = 0
total_frames = 0
frame_num = 0
images_list = []
# queue = []
while (True):
    ret, image = carVideo.read()
    # print("image shape",image.shape)

    if ret == False:
        print("not possible")
        break
    if frame_num % 25 == 0: # picking after every 25 frame
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 320))
        img = img.astype(np.float32)
        img = np.expand_dims(img, 0)
        img = img / 255
        class_names = [c.strip() for c in open("E:\python projects\Vehicle-Detection-Classification-YOLO-MobileNet-main\classes.TXT").readlines()]

        boxes, scores, classes, nums = yolo(img) # detecting cars in yolov3

        # count = 0
        for i in range(nums[0]):
            if int(classes[0][i] == 2):  # car in image
                print("is a car")
                # count += 1
            elif int(classes[0][1] == 5):
                print("is a bus")
            elif int(classes[0][1] == 7):
                print("is a truck")

        # video edit of cars bounding box
        draw_image = Image.fromarray(image)
        classes1 = np.array(classes).astype("int32")
        # boxes1 = np.array(boxes)
        # scores1 = np.array(scores)

        for i, c in enumerate(classes1): # the list was reversed but i removed it
        #     if c[i] == 0:
        #         break
        #     ui = c[i]
        #     predicted_class = class_names[c[i]]
        #     box = boxes1[i][i]
        #     score = scores1[i][i]
        #
        #     label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(draw_image)
            # label_size = draw.textsize(label, font)

            # top, left, bottom, right = box
            # top = max(0, np.floor(top + 0.1).astype('int32'))
            # left = max(0, np.floor(left + 0.1).astype('int32'))
            # bottom = min(image.size[1], np.floor(bottom + 0.1).astype('int32'))
            # right = min(image.size[0], np.floor(right + 0.1).astype('int32'))
            # print(label, (left, top), (right, bottom))

            # position = dict()
            # position["class"] = predicted_class
            # position["left"] =  left
            # position["top"] = top
            # position["right"] = right
            # position["bottom"] = bottom

            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            # else:
            #     text_origin = np.array([left, top + 1])
            hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]

            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

            colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

            thickness = (draw_image.size[0] + draw_image.size[1]) // 800
            # My kingdom for a good redistributable image drawing library.
            # for i in range(thickness):
            #     draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
            # boxes, objectness, classes, nums = outputs
            boxes1, objectness, classes1, nums1 = boxes[0], scores[0], classes[0], nums[0]
            # x, y, w, h = boxes1
            # x = int(x - w / 2)
            # y = int(y - h / 2)
            # w = int(w)
            # h = int(h)
            # cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
            wh = np.flip(image.shape[0:2])
            # wh = image.shape[0:2]
            position_list = []
            for j in range(nums1):
                x1y1 = tuple((np.array(boxes1[j][0:2]) * wh).astype(np.int32))
                x2y2 = tuple((np.array(boxes1[j][2:4]) * wh).astype(np.int32))
                # position_list
            # for k in range(thickness):
            #     draw.rectangle([x1y1[1], x2y2[1], x1y1[0],x2y2[0]], outline=colors[c[j]])
            #     draw.rectangle([x1y1[0], x2y2[0], x1y1[1],x2y2[1]], outline=colors[c[j]])
            #     draw.rectangle([x1y1[0],x2y2[1], x1y1[1], x2y2[0]], outline=colors[c[j]])
                draw.rectangle([x2y2[0], x2y2[1], x1y1[0], x1y1[1]], outline=colors[c[j]])


            # imgplot = plt.imshow(draw_image)
            # plt.show()

            # queue.append(frame_num)

            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=self.colors[c])
            # draw.text(text_origin, label, fill=(0, 0, 0), font=font)

            imgplot = plt.imshow(draw_image)
            plt.show()
        images_list.append(draw_image)
        out_image = draw_outputs(image, (boxes, scores, classes, nums), class_names)
        print("frame number = ",frame_num)
        total_frames += 1
    frame_num += 1

#     image_size = image.shape[0:2]
#     # image_size = image.shape
# print("total frames = ", total_frames)
# video = cv2.VideoWriter('MY_Output_NEW_1.avi', cv2.VideoWriter_fourcc(*"MJPG"), 1, image_size)  # saving output video
#
# for i ,image_in_list in enumerate(images_list):
#
#
#     # for i in range(1, len(images_list) + 1):
#         # q, image = self.detectedCars[i]
#
#     np_image = np.asarray(image_in_list)
#     # for position in q:
#     #     # pos,pred = car.get_position(),car.get_carType()
#     #     color = (0, 0, 255)
#         # if pred=="SUV":
#         #     color = (0,170,230)
#         # cv2.putText(np_image,str(pred),pos,cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
#     position = (10, 20)
#     cv2.putText(np_image, "Result Video", position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
#                 2)  # font stroke
#     position = (10, 40)
#     cv2.putText(np_image, "Frame : " + str(i), position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
#                 2)  # font stroke
#     position = (10, 60)
#     cv2.putText(np_image, "Count Car : " + str(counter), position, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                 (0, 0, 255), 2)  # font stroke
#     result = np.asarray(np_image)
#     # print("result shape",result.shape)
#     video.write(result)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# video.release()
#
#
#     # video edit of cars bounding box
#     # cv2.imshow('Prediction', image)
#
#
#
#
# # if cv2.waitKey(1) & 0xFF == ord('s'):
# #     break
# counter += 1
#
# carVideo.release()
# cv2.destroyAllWindows()
