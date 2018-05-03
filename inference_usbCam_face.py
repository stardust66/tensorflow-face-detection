#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2

from utils import label_map_util
from utils import visualization_utils_color as vis_util

PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class TensorflowFaceDetector():
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector

        Args:
        PATH_TO_CKPT: path to saved model checkpoint.
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            old_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as graph_file:
                serialized_graph = graph_file.read()
                old_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(old_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(graph=self.detection_graph, config=config) as self.sess:

                self.windowNotSet = True


    def run(self, image):
        """image: bgr image

        return (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Expand dimensions since the model expects images to have shape:
        # [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        input_image = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={input_image: image_np_expanded}
        )

        return (boxes, scores, classes, num_detections)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print """usage:%s (cameraID | filename)
Detect faces in the video
example:
%s 0
""" % (sys.argv[0], sys.argv[0])
        exit(1)

    try:
        camID = int(sys.argv[1])
    except:
        camID = sys.argv[1]

    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    cap = cv2.VideoCapture(camID)
    windowNotSet = True
    while True:
        ret, image = cap.read()
        if ret == 0:
            break

        [h, w] = image.shape[:2]
        print h, w
        image = cv2.flip(image, 1)

        (boxes, scores, classes, num_detections) = tDetector.run(image)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4)

        if windowNotSet is True:
            cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
            windowNotSet = False

        cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
