# from ultralytics import YOLO
# import yolov8

# model = YOLO("E:\python projects\Vehicle-Detection-Classification-YOLO-MobileNet-main\yolov8n.pt")
#
# model.predict(source="E:\python projects\Vehicle-Detection-Classification-YOLO-MobileNet-main\test\carnew1.mp4", save=False, conf = 0.5, save_text = False, show = True)



# import cv2
# import darknet
#
# net = darknet.load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
# meta = darknet.load_meta("cfg/coco.data")
#
# img = cv2.imread("image.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# height, width = img.shape[:2]
#
# if label == "car":
#         x, y, w, h = bbox
#         x = int(x - w / 2)
#         y = int(y - h / 2)
#         w = int(w)
#         h = int(h)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)