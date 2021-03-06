import numpy as np
import sys
import os
import tensorflow as tf
import cv2


class ObjectDetector:

  def __init__(self, config, threshold=0.5):
    assert config.TF_MODEL_FOLDER is not None
    # to be able to import tensorflow models
    sys.path.append(config.TF_MODEL_FOLDER)
    from object_detection.utils import label_map_util
    # from object_detection.utils import visualization_utils as vis_util
    # self.vis_util = vis_util
    self.label_map_util = label_map_util

    self.config = config
    self.threshold = threshold
    self.MODEL_NAME = "faster_rcnn_resnet101_coco_11_06_2017"
    self.PATH_TO_CKPT = os.path.join(config.TF_MODEL_OD_CKPT_FOLDER,
      self.MODEL_NAME + "/frozen_inference_graph.pb")
    # self.PATH_TO_LABELS = os.path.join(config.TF_MODEL_FOLDER,
    #   "object_detection/data/mscoco_label_map.pbtxt")
    self.PATH_TO_LABELS = config.COCO_LABEL_MAP_PATH
    self.NUM_CLASSES = 90
    self.isInitialized = False

  def initialize(self):
    self.isInitialized = True
    self.loadModel()
    self.loadLabelMap()

  def loadModel(self):
    print(">>> load model pb file: {}".format(self.PATH_TO_CKPT))
    self.detection_graph = tf.Graph()
    with self.detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(self.PATH_TO_CKPT, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")

  def loadLabelMap(self):
    """Label maps map indices to category names,

    so that when our convolution network predicts 5,
    we know that this corresponds to airplane.
    Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    """
    print(">>> load Label Map: {}".format(self.PATH_TO_LABELS))
    self.label_map = self.label_map_util.load_labelmap(
      self.PATH_TO_LABELS)
    self.categories = self.label_map_util.convert_label_map_to_categories(
      self.label_map,
      max_num_classes=self.NUM_CLASSES,
      use_display_name=True)

    self.category_index = self.label_map_util.create_category_index(
      self.categories)
    # create a list filled with 0
    self.categoryNames = [0 for x in range(self.NUM_CLASSES)]
    for cc in self.categories:
      self.categoryNames[int(cc["id"]) - 1] = cc["name"]

  def parse(self, images):
    if not self.isInitialized:
      self.initialize()

    results = []
    with self.detection_graph.as_default():
      with tf.Session(graph=self.detection_graph) as sess:
        image_tensor = self.detection_graph.get_tensor_by_name(
          "image_tensor:0")
        # Each box represents a part of the image where a particular object was detected.
        boxes_ = self.detection_graph.get_tensor_by_name(
          "detection_boxes:0")
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores_ = self.detection_graph.get_tensor_by_name(
          "detection_scores:0")
        classes_ = self.detection_graph.get_tensor_by_name(
          "detection_classes:0")
        num_detections_ = self.detection_graph.get_tensor_by_name(
          "num_detections:0")

        for image in images:
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_expanded = np.expand_dims(image, axis=0)
          boxes, scores, classes, num_detections = sess.run(
            [boxes_, scores_, classes_, num_detections_],
            feed_dict={image_tensor: image_expanded})
          indices = np.where(np.squeeze(scores) > self.threshold)
          scores = np.squeeze(scores)[indices].tolist()
          boxes = np.squeeze(boxes)[indices].tolist()
          classes = np.squeeze(classes).astype(np.int32)[indices]
          # convert it to names
          classes = [self.categoryNames[cc - 1] for cc in classes]
          print("classes: {}".format(classes))
          num_detections = len(classes)
          results.append((boxes, scores, classes, num_detections))
          # visualization
          # if __name__ == "__main__":
          #   print("should open visualization")
          #   self.vis_util.visualize_boxes_and_labels_on_image_array(
          #     image,
          #     np.squeeze(boxes),
          #     np.squeeze(classes).astype(np.int32),
          #     np.squeeze(scores),
          #     self.category_index,
          #     use_normalized_coordinates=True,
          #     line_thickness=8)

        return results

if __name__ == "__main__":
  from types import SimpleNamespace
  config = SimpleNamespace()
  config.TF_MODEL_FOLDER = "/Users/eisneim/www/deepLearning/_tf_models"
  config.TF_MODEL_OD_CKPT_FOLDER = "/Volumes/raid/_deeplearning/_models/tf_detection_modelzoo/"
  config.COCO_LABEL_MAP_PATH = "data/mscoco_label_map_cn.pbtxt"
  od = ObjectDetector(config)
  od.initialize()

  img1 = cv2.imread("testData/ocr1.png")
  img2 = cv2.imread("testData/img2.jpg")
  boxes, scores, classes, num_detections = od.detect([img1, img2])[0]
  # print("od.categories:", od.categoryNames)

  print("num_detections", num_detections)
  print("boxes", boxes)
  print("classes", classes)
  print("scores:", scores)


