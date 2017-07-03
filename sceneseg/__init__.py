import cv2
import numpy as np


class SceneSeg(object):
  """Base sc_detector class to implement a scene detection algorithm."""
  def __init__(self):
    pass

  def process_frame(self, frameNum, frameImg, frameMetrics, sceneList):
    """Computes/stores metrics and detects any scene changes.
    Prototype method, no actual detection.
    """
    return

  def post_process(self, sceneList):
    pass




