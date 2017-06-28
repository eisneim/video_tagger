import cv2
import numpy as np
import os
import logging

from utils.util_img import ratio_scale_factor

log = logging.getLogger("vtr")


class VideoParserState:
  def __init__(self, donwScaleFactor=1):
    self._currentFrame = None
    self.frameNum = 0
    self.scenes = []
    self.donwScaleFactor = donwScaleFactor

  @property
  def currentFrame(self):
    return self._currentFrame

  @property.setter
  def currentFrame(self, frame):
    self._currentFrame = frame
    self.scaledFrame = frame[::self.downscale_factor, ::self.downscale_factor, :]


class VideoTagger:

  def __init__(self, config):
    self.config = config
    self.state = VideoParserState()

  def parse(self, videoPath):

    cap = cv2.VideoCapture

    # @TODO: caught errors
    cap.open(videoPath)
    fileName = os.path.split(videoPath)[1]
    if not cap.isOpened():
      log.critical("Could not open video: {}".format(fileName))
      return "Unable to open video", None

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
      log.info("donwScaleFactor: {}".format(self.donwScaleFactor))

    # main loop
    self.loop(cap)

    # Cleanup and return parse state.
    cap.release()
    return None, self.state

  def loop(self, cap):
    self.state.frameNum = 0
    while cap.isOpened():
      _, frame = cap.read()
      self.state.currentFrame = frame
      # this is where calculation happens
      self.tick(frame)
      self.state.frameNum += 1

  def tick(self, frame):
    pass
