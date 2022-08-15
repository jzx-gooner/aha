#!/usr/bin/env python3

import numpy as np
import cv2
import colorsys
import time

# Camera API libs
from hobot_vio import libsrcampy as srcampy
from hobot_dnn import pyeasy_dnn as dnn

# detection model class names
def get_classes():
    return np.array(["person", "bicycle", "car",
                     "motorcycle", "airplane", "bus",
                     "train", "truck", "boat",
                     "traffic light", "fire hydrant", "stop sign",
                     "parking meter", "bench", "bird",
                     "cat", "dog", "horse",
                     "sheep", "cow", "elephant",
                     "bear", "zebra", "giraffe",
                     "backpack", "umbrella", "handbag",
                     "tie", "suitcase", "frisbee",
                     "skis", "snowboard", "sports ball",
                     "kite", "baseball bat", "baseball glove",
                     "skateboard", "surfboard", "tennis racket",
                     "bottle", "wine glass", "cup",
                     "fork", "knife", "spoon",
                     "bowl", "banana", "apple",
                     "sandwich", "orange", "broccoli",
                     "carrot", "hot dog", "pizza",
                     "donut", "cake", "chair",
                     "couch", "potted plant", "bed",
                     "dining table", "toilet", "tv",
                     "laptop", "mouse", "remote",
                     "keyboard", "cell phone", "microwave",
                     "oven", "toaster", "sink",
                     "refrigerator", "book", "clock",
                     "vase", "scissors", "teddy bear",
                     "hair drier", "toothbrush"])


def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12


def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


def postprocess(model_output,
                model_hw_shape,
                origin_image=None,
                origin_img_shape=None,
                score_threshold=0.5,
                nms_threshold=0.6,
                dump_image=False):
    input_height = model_hw_shape[0]
    input_width = model_hw_shape[1]
    if origin_image is not None:
        origin_image_shape = origin_image.shape[0:2]
    else:
        origin_image_shape = origin_img_shape

    prediction_bbox = decode(outputs=model_output,
                             score_threshold=score_threshold,
                             origin_shape=origin_image_shape,
                             input_size=512)

    prediction_bbox = nms(prediction_bbox, iou_threshold=nms_threshold)

    prediction_bbox = np.array(prediction_bbox)
    topk = min(prediction_bbox.shape[0], 1000)

    if topk != 0:
        idx = np.argpartition(prediction_bbox[..., 4], -topk)[-topk:]
        prediction_bbox = prediction_bbox[idx]

    if dump_image and origin_image is not None:
        draw_bboxs(origin_image, prediction_bbox)
    return prediction_bbox


def draw_bboxs(image, bboxes, gt_classes_index=None, classes=get_classes()):
    """draw the bboxes in the original image
    """
    num_classes = len(classes)
    image_h, image_w, channel = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))

    fontScale = 0.5
    bbox_thick = int(0.6 * (image_h + image_w) / 600)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)

        if gt_classes_index == None:
            class_index = int(bbox[5])
            score = bbox[4]
        else:
            class_index = gt_classes_index[i]
            score = 1

        bbox_color = colors[class_index]
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        #time.sleep(8)
        #c_time_path ="./images/"+str(time.time())+".jpg"
        #cv2.imwrite(c_time_path,image)
        #cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        #classes_name = classes[class_index]
        #bbox_mess = '%s: %.2f' % (classes_name, score)
        #t_size = cv2.getTextSize(bbox_mess,
        #                         0,
        #                         fontScale,
        #                         thickness=bbox_thick // 2)[0]
        #cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3),
        #              bbox_color, -1)
        #cv2.putText(image,
        #            bbox_mess, (c1[0], c1[1] - 2),
        #            cv2.FONT_HERSHEY_SIMPLEX,
        #            fontScale, (0, 0, 0),
        #            bbox_thick // 2,
        #            lineType=cv2.LINE_AA)
        #print("{} is in the picture with confidence:{:.4f}, bbox:{}".format(
        #    classes_name, score, coor))
    #    cv2.imwrite("demo.jpg", image)
    return image


def decode(outputs, score_threshold, origin_shape, input_size=512):
    def _distance2bbox(points, distance):
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]
        return np.stack([x1, y1, x2, y2], -1)

    def _scores(cls, ce):
        cls = 1 / (1 + np.exp(-cls))
        ce = 1 / (1 + np.exp(-ce))
        return np.sqrt(ce * cls)

    def _bbox(bbox, stride, origin_shape, input_size):
        h, w = bbox.shape[1:3]
        yv, xv = np.meshgrid(np.arange(h), np.arange(w))
        xy = (np.stack((yv, xv), 2) + 0.5) * stride
        bbox = _distance2bbox(xy, bbox)
        # opencv read, shape[1] is w, shape[0] is h
        scale_w = origin_shape[1] / input_size
        scale_h = origin_shape[0] / input_size
        scale = max(origin_shape[0], origin_shape[1]) / input_size
        # origin img is pad resized
        #bbox = bbox * scale
        # origin img is resized
        bbox = bbox * [scale_w, scale_h, scale_w, scale_h]
        return bbox

    bboxes = list()
    strides = [8, 16, 32, 64, 128]

    for i in range(len(strides)):
        cls = outputs[i].buffer
        bbox = outputs[i + 5].buffer
        ce = outputs[i + 10].buffer
        scores = _scores(cls, ce)

        classes = np.argmax(scores, axis=-1)
        classes = np.reshape(classes, [-1, 1])
        max_score = np.max(scores, axis=-1)
        max_score = np.reshape(max_score, [-1, 1])
        bbox = _bbox(bbox, strides[i], origin_shape, input_size)
        bbox = np.reshape(bbox, [-1, 4])

        pred_bbox = np.concatenate([bbox, max_score, classes], axis=1)

        index = pred_bbox[..., 4] > score_threshold
        pred_bbox = pred_bbox[index]
        bboxes.append(pred_bbox)

    return np.concatenate(bboxes)


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    def bboxes_iou(boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
                      (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
                      (boxes2[..., 3] - boxes2[..., 1])
        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area,
                          np.finfo(np.float32).eps)

        return ious

    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)

if __name__ == '__main__':
    models = dnn.load('../models/fcos_512x512_nv12.bin')
    # 打印输入 tensor 的属性
    print_properties(models[0].inputs[0].properties)
    # 打印输出 tensor 的属性
    print(len(models[0].outputs))
    for output in models[0].outputs:
        print_properties(output.properties)

    # Camera API, get camera object
    cam = srcampy.Camera()
    # Open f37 camera
    # For the meaning of parameters, please refer to the relevant documents of camera
    h, w = get_hw(models[0].inputs[0].properties)
    cam.open_cam(0, 1, 30, w, h)

    # Get HDMI display object
    disp = srcampy.Display()
    # For the meaning of parameters, please refer to the relevant documents of HDMI display
    disp.display(0, 1920, 1080)
    input_shape = (h, w)
    while True:
        # Get image data with shape of 512x512 nv12 data from camera
        img = cam.get_img(2, 512, 512)
        # Convert to numpy
        img = np.frombuffer(img, dtype=np.uint8)
        # Forward
        outputs = models[0].forward(img)
        # Do post process
        prediction_bbox = postprocess(outputs, input_shape, origin_img_shape=(1080,1920))

        # Get origin img with shape of 1920x1080(F37) nv12 data for HDMI display
        origin_image = cam.get_img(2, 1920, 1080)
        origin_nv12 = np.frombuffer(origin_image, dtype=np.uint8).reshape(1620, 1920)
        origin_bgr = cv2.cvtColor(origin_nv12, cv2.COLOR_YUV420SP2BGR)
        origin_bgr_1 = cv2.resize(origin_bgr,(640,480))
        kk = cv2.waitKey(2)
        #if kk == ord('Q') or kk == ord('q'):
            #print("**********")
        time.sleep(8)
        c_time_path ="./images/"+str(time.time())+".jpg"
        cv2.imwrite(c_time_path,origin_bgr_1)

        # Draw bboxs
        #box_bgr = draw_bboxs(origin_bgr, prediction_bbox)
        # Convert to nv12 for HDMI display
        box_nv12 = bgr2nv12_opencv(origin_bgr)

        disp.set_img(box_nv12.tobytes())   
    cam.close_cam()
