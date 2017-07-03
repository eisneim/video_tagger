import cv2
import numpy as np
import os
import logging

from utils.util_img import ratio_scale_factor
from sceneseg.content_detector import ContentDetector

log = logging.getLogger("vtr")


class VideoParserState(object):
  def __init__(self, donwScaleFactor=1, predDonwScaleFactor=1):
    self._currentFrame = None
    self.frameNum = 0
    self.lastSceneFrameNum = 0
    self.scenes = []
    # only store scaled frame to save memory space
    self.allScaledFrames = []
    # buffer of frames, if new scene is detected
    # previous scene frames will be discard.
    # those frames are used to properly represent
    # a scene
    self.currentSceneFrames = []
    self.donwScaleFactor = donwScaleFactor
    self.predDonwScaleFactor = predDonwScaleFactor
    print("donwScaleFactor: {}, predDonwScaleFactor: {}".format(
      donwScaleFactor,
      predDonwScaleFactor))

  @property
  def currentFrame(self):
    return self._currentFrame

  @currentFrame.setter
  def currentFrame(self, frame):
    self._currentFrame = frame
    self.scaledFrame = frame
    if self.donwScaleFactor > 1:
      self.scaledFrame = frame[::self.donwScaleFactor, ::self.donwScaleFactor, :]

    if self.predDonwScaleFactor > 1:
      predFrame = frame[::self.predDonwScaleFactor, ::self.predDonwScaleFactor, :]
      self.currentSceneFrames.append(predFrame)
    else:
      self.currentSceneFrames.append(frame)

    self.allScaledFrames.append(self.scaledFrame)


class VideoTagger:
  """add tags to video scnenes

  perform scene segmentation; for each scene, find all tags to describe
  this scene, those tags can be used for video searching

  Variables:
    config {object} -- [description]
    config.max_cal_width {number} -- max frame width for frame differience
    config.max_cal_height {number} -- [description]
    config.max_pred_width {number} -- max frame width for object detection
    config.max_pred_height {number} -- [description]
    config.frame_skip {number} -- skip frames to get faster performance
  """
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

    # donw-scale frame size for object detection
    if self.config.max_pred_width:
      self.config.max_pred_width = int(self.config.max_pred_width)
      self.config.max_pred_height = int(self.config.max_pred_height)
      scaleFactor = ratio_scale_factor(self.videoWidth,
        self.videoHeight,
        self.config.max_pred_width,
        self.config.max_pred_height)
      print("scaleFactor:", scaleFactor, round(1 / scaleFactor))
      self.predDonwScaleFactor = round(1 / scaleFactor)
      log.info("predDonwScaleFactor: {}".format(self.donwScaleFactor))

    # store parsing state
    self.state = VideoParserState(self.donwScaleFactor,
      self.predDonwScaleFactor)

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
    # used for storing each frame's metrics
    # those metrics params depending on which detector you use.
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
    # frameMatrics
    if self.state.frameNum not in self.frameMetrics:
      self.frameMetrics[self.state.frameNum] = {}

    isCutFound = self.detector.process_frame(
      self.state.frameNum,
      frame,
      self.frameMetrics,
      self.sceneList)

    if isCutFound:
      self.sceneDetected()
      print("cut found at: {}".format(self.state.frameNum))

  def sceneDetected(self):
    self.state.scenes.append(self.state.frameNum)
    # should find best frame to represent this scene
    # currently using the dumbest way, which is using the middle frame
    # @TODO: better way to find representative frame
    midIndex = len(self.state.currentSceneFrames) // 2
    dominateFrame = self.state.currentSceneFrames[midIndex]
    # empty scenFrames buffer
    self.state.currentSceneFrames = []
    # >>> should start object detection on dominateFrame



if __name__ == "__main__":
  from types import SimpleNamespace
  config = SimpleNamespace()
  config.max_cal_width = "320"
  config.max_cal_height = "480"
  config.max_pred_width = 640
  config.max_pred_height = 720
  config.frame_skip = 2

  tagger = VideoTagger(config)
  testVdieo = "testData/video.mp4"
  err, state = tagger.parse(testVdieo)
  print("err: ", err)
  print("allScaledFrames: {}, shape:{}".format(
    len(state.allScaledFrames),
    state.allScaledFrames[0].shape))
  print("currentSceneFrames: {}, shape:{}".format(
    len(state.currentSceneFrames),
    state.currentSceneFrames[0].shape))
  print("scenes: {}".format(state.scenes))
  print("self.sceneList: ", tagger.sceneList)


