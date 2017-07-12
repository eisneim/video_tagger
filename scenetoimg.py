"""
convert video to images, based on scene cut
it will try to save the first and last frame of a scene
"""
import cv2
import numpy as np
import os
import sys
import datetime

from sceneseg.content_detector import ContentDetector
from utils.util_img import ratio_scale_factor


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
    self.objectDetectionResults = []
    self.captioningResults = []

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


class VideoToImg:
  def __init__(self, config, destFolder):
    self.config = config
    self.destFolder = destFolder
    self.scDetector = ContentDetector(
      threshold=config.THRESHOLD_SCENESEG,
      minFrames=config.SCENESEG_MINFRAMES)

  def parse(self, videoPath):
    self.videoName = os.path.basename(videoPath).rsplit(".")[0]
    cap = cv2.VideoCapture()

    # @TODO: caught errors
    cap.open(videoPath)
    if not cap.isOpened():
      fileName = os.path.split(videoPath)[1]
      print("Could not open video: {}".format(fileName))
      return "Unable to open video", None

    self.videoWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    self.videoHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    self.videoFps = cap.get(cv2.CAP_PROP_FPS)
    # perform the down-scale step
    donwScaleFactor = 1
    predDonwScaleFactor = 1
    if self.videoWidth > self.config.max_cal_width or \
      self.videoHeight > self.config.max_cal_height:
      scaleFactor = ratio_scale_factor(self.videoWidth,
        self.videoHeight,
        self.config.max_cal_width,
        self.config.max_cal_height)
      donwScaleFactor = round(1 / scaleFactor)
      print("donwScaleFactor: {}".format(donwScaleFactor))

    # donw-scale frame size for object detection
    if self.videoWidth > self.config.max_pred_width or \
      self.videoHeight > self.config.max_pred_height:
      scaleFactor = ratio_scale_factor(self.videoWidth,
        self.videoHeight,
        self.config.max_pred_width,
        self.config.max_pred_height)
      print("scaleFactor: {} {}".format(scaleFactor, round(1 / scaleFactor)))
      predDonwScaleFactor = round(1 / scaleFactor)
      print("predDonwScaleFactor: {}".format(donwScaleFactor))

    # store parsing state
    self.state = VideoParserState(donwScaleFactor,
      predDonwScaleFactor)

    # main loop
    self.loop(cap)

    # Cleanup and return parse state.
    cap.release()
    return None, self.state

  def loop(self, cap):
    skip = 0
    if self.config.frame_skip and self.config.frame_skip > 0:
      skip = self.config.frame_skip

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
        # from last cut to the last frame number
        # there is still a scene to be parsed
        self.sceneDetected(isLastFrame=True)
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

    isCutFound = self.scDetector.process_frame(
      self.state.frameNum,
      frame,
      self.frameMetrics,
      self.sceneList)

    if isCutFound:
      self.sceneDetected()
      print("cut found at: {}".format(self.state.frameNum))

  def sceneDetected(self, isLastFrame=False):
    if not isLastFrame:
      self.state.scenes.append(self.state.frameNum)
    # should find best frame to represent this scene
    # currently using the dumbest way, which is using the middle frame
    # @TODO: better way to find representative frame
    frameCount = len(self.state.currentSceneFrames)
    midIndex = frameCount // 2
    firstFrame = self.state.currentSceneFrames[0]
    midFrame = self.state.currentSceneFrames[midIndex]
    lastFrame = self.state.currentSceneFrames[-1]
    # @TODO: should we save middle frame?
    framesToBeSaved = [firstFrame,
      # midFrame,
      lastFrame]
    framesNumbers = [self.state.frameNum - frameCount,
      # self.state.frameNum - midIndex,
      self.state.frameNum]

    # empty scenFrames buffer
    self.state.currentSceneFrames = []

    # save images
    datestr = datetime.datetime.now().strftime("%m-%d_%H_%M%S")
    for idx, img in enumerate(framesToBeSaved):
      frameNum = framesNumbers[idx]
      name = "{}_{}_{}.jpg".format(datestr, self.videoName, frameNum)
      path = os.path.join(self.destFolder, name)
      cv2.imwrite(path, img)
      print("save image: {}".format(path))


def main():
  sourceVideo = sys.argv[1]
  destFolder = sys.argv[2]
  from types import SimpleNamespace
  config = SimpleNamespace()
  config.max_cal_width = 320
  config.max_cal_height = 480
  config.max_pred_width = 640
  config.max_pred_height = 720
  config.frame_skip = 2
  config.THRESHOLD_SCENESEG = 30
  config.SCENESEG_MINFRAMES = 15


  parser = VideoToImg(config, destFolder)
  err, state = parser.parse(sourceVideo)

if __name__ == "__main__":
  main()

