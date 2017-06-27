import cv2
import numpy as np
import os

from utils.util_img import ratio_scale_factor


class VideoParserState:
  def __init__(self):
    self.currentFrame = None
    self.scenes = []


class VideoTagger:

  def __init__(self, config):
    self.config = config
    self.state = VideoParserState()

  def commence(self, videoPath):

    cap = cv2.VideoCapture
    # @TODO: caught errors
    cap.open(videoPath)
    fileName = os.path.split(videoPath)[1]
    if not cap.isOpened():
      log.critical("Could not open video: {}".format(fileName))

    self.videoWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    self.videoHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    self.videoFps = cap.get(cv2.CAP_PROP_FRAME_FPS)
    # perform the down-scale step
    if self.config.max_cal_width:
      self.config.max_cal_width = int(self.config.max_cal_width)
      self.config.max_cal_height = int(self.config.max_cal_height)
      scaleFactor = ratio_scale_factor(self.videoWidth,
        self.videoHeight,
        self.config.max_cal_width,
        self.config.max_cal_height)
      self.donwScaleFactor = round(1 / scaleFactor)


  def step(self):
    pass
