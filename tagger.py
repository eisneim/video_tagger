import cv2
import numpy as np
import os
import logging

from utils.util_img import ratio_scale_factor
from sceneseg.content_detector import ContentDetector

log = logging.getLogger("vtr")


class VideoParserState(object):
  def __init__(self, donwScaleFactor=1):
    self._currentFrame = None
    self.frameNum = 0
    self.lastSceneFrameNum = 0
    self.scenes = []
    self.downscale_factor = 1
    # only store scaled frame to save memory space
    self.allScaledFrames = []
    # buffer of frames, if new scene is detected
    # previous scene frames will be discard.
    # those frames are used to properly represent
    # a scene
    self.currentSceneFrames = []
    self.donwScaleFactor = donwScaleFactor

  @property
  def currentFrame(self):
    return self._currentFrame

  @currentFrame.setter
  def currentFrame(self, frame):
    self._currentFrame = frame
    self.scaledFrame = frame
    if self.downscale_factor > 1:
      self.scaledFrame = frame[::self.downscale_factor, ::self.downscale_factor, :]

    self.allScaledFrames.append(self.scaledFrame)
    self.currentSceneFrames.append(frame)

  def sceneDetected(self):


    # should find best frame to represent this scene

    # empty scenFrames buffer
    self.currentSceneFrames = []


class VideoTagger:

  def __init__(self, config):
    self.config = config
    self.detector = ContentDetector(threshold=30, minFrames=15)

  def parse(self, videoPath):

    cap = cv2.VideoCapture()

    # @TODO: caught errors
    cap.open(videoPath)
    fileName = os.path.split(videoPath)[1]
    if not cap.isOpened():
      log.critical("Could not open video: {}".format(fileName))
      return "Unable to open video", None

    self.videoWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    self.videoHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    self.videoFps = cap.get(cv2.CAP_PROP_FPS)
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

    # store parsing state
    self.state = VideoParserState(self.donwScaleFactor)

    # main loop
    self.loop(cap)

    # Cleanup and return parse state.
    cap.release()
    return None, self.state

  def loop(self, cap):
    skip = 0
    if self.config.frame_skip and self.config.frame_skip > 0:
      skip = int(self.config.frame_skip)

    self.state.frameNum = 0
    self.frameMetrics = {}
    self.sceneList = []

    while cap.isOpened():
      # frame skip is set, should drop corespond frames
      if skip > 0:
        for _ in range(skip):
          ret_val = cap.grab()
          if not ret_val:
            break
          self.state.frameNum += 1

      ret_val, frame = cap.read()
      if not ret_val:
        break

      # save current frame
      self.state.currentFrame = frame
      # this is where calculation happens
      self.tick(frame)
      self.state.frameNum += 1

  def tick(self, frame):

    if self.state.frameNum not in self.frameMetrics:
      self.frameMetrics[self.state.frameNum] = {}

    isCutFound = self.detector.process_frame(
      self.state.frameNum,
      frame,
      self.frameMetrics,
      self.sceneList)

    if isCutFound:
      self.state.sceneDetected()
      print("cut found at: {}".format(self.state.frameNum))


if __name__ == "__main__":
  from types import SimpleNamespace
  config = SimpleNamespace()
  config.max_cal_width = "320"
  config.max_cal_height = "480"
  config.frame_skip = 2

  tagger = VideoTagger(config)
  testVdieo = "testData/video.mp4"
  err, state = tagger.parse(testVdieo)
  print("err: ", err)
  print("allScaledFrames: {}".format(len(state.allScaledFrames)))
  print("currentSceneFrames: {}".format(len(state.currentSceneFrames)))
  print("scenes: {}".format(state.scenes))
  print("self.sceneList: ", tagger.sceneList)


