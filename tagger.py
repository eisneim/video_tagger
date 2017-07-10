import cv2
import numpy as np
import os
import sys
import logging
import time
from multiprocessing import Process, Manager, current_process

from utils.util_img import ratio_scale_factor
from utils.util_vis import drawBoxes
from sceneseg.content_detector import ContentDetector
from object_detector import ObjectDetector
import batch_im2txt

log = logging.getLogger("vtr")
PROCESS_TIMEOUT = 15

PROCESS_OD = "PROCESS_OD"
PROCESS_IM2TXT = "PROCESS_IM2TXT"

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
    log.info("donwScaleFactor: {}, predDonwScaleFactor: {}".format(
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
    self.scDetector = ContentDetector(threshold=30, minFrames=15)
    qManager = Manager()
    self.inputDict = qManager.dict()
    self.outputDict = qManager.dict()
    # self.queue = Queue()

    # create neural network process
    self.process_od = Process(target=process_pooling,
      args=(self.inputDict, self.outputDict, PROCESS_OD, config),
      name=PROCESS_OD)
    self.process_od.daemon = True

    self.process_im2txt = Process(target=process_pooling,
      args=(self.inputDict, self.outputDict, PROCESS_IM2TXT, config),
      name=PROCESS_IM2TXT)
    self.process_im2txt.daemon = True

    self.processes = [PROCESS_OD, PROCESS_IM2TXT]
    # initialize process dict
    for pid in self.processes:
      self.inputDict[pid] = None
      self.outputDict[pid] = None
      self.outputDict[pid + "_INIT"] = None
    # start all child processes
    self.process_im2txt.start()
    self.process_od.start()

    self.allChildProcessReady = False

  def blockUntilChildprocessReady(self):
    log.info(">> waiting for all child process to be ready")
    while True:
      count = 0
      for pid in self.processes:
        isReady = self.outputDict[pid + "_INIT"]
        count += 1 if isReady else 0
      if count == len(self.processes):
        self.allChildProcessReady = True
        break

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
      log.info("scaleFactor: {} {}".format(scaleFactor, round(1 / scaleFactor)))
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

    isCutFound = self.scDetector.process_frame(
      self.state.frameNum,
      frame,
      self.frameMetrics,
      self.sceneList)

    if isCutFound:
      self.sceneDetected()
      log.info("cut found at: {}".format(self.state.frameNum))

  def sceneDetected(self):
    if not self.allChildProcessReady:
      self.blockUntilChildprocessReady()

    self.state.scenes.append(self.state.frameNum)
    # should find best frame to represent this scene
    # currently using the dumbest way, which is using the middle frame
    # @TODO: better way to find representative frame
    midIndex = len(self.state.currentSceneFrames) // 2
    dominateFrame = self.state.currentSceneFrames[midIndex]
    # empty scenFrames buffer
    self.state.currentSceneFrames = []
    # >>> should start object detection on dominateFrame
    # boxes, scores, classes, num_detections
    # detectedResults = self.objDetector.parse(
    #   [dominateFrame])
    self.inputDict[PROCESS_OD] = [dominateFrame]
    self.inputDict[PROCESS_IM2TXT] = [dominateFrame]

    # start to pool results from queue:
    resultCount = 0
    resultData = {}
    startTime = time.time()
    while resultCount < len(self.processes):
      for identifier in self.processes:
        data = self.outputDict[identifier]
        if data is not None:
          resultCount += 1
          self.outputDict[identifier] = None
        resultData[identifier] = data

      if time.time() - startTime > PROCESS_TIMEOUT:
        # should kill all process
        raise ValueError("PROCESS_TIMEOUT")

    # visualize boundingboxes
    if not config.IS_PRODUCTION:
      for result in resultData[PROCESS_OD]:
        boxes, scores, classes, _ = result
        img = drawBoxes(dominateFrame.copy(), boxes, scores, classes)

      cv2.imshow("detection", img)
      cv2.waitKey(5)

    # should save the frame(s) to filesystem


# def createProcess(dd, queue, identifier):
#   if identifier == PROCESS_OD:
#     return


def process_pooling(inputDict, outputDict, identifier, config):
  name = current_process().name
  # make print() to a file
  # sys.stdout = open("log/" + name + ".out", "a")
  # sys.stderr = open("log/" + name + "_error.out", "a")
  log.info("start process: {}".format(name))

  targetProcess = None
  if identifier == PROCESS_OD:
    targetProcess = ObjectDetector(config)
    # build the network
    targetProcess.initialize()
    outputDict[identifier + "_INIT"] = True

  if identifier == PROCESS_IM2TXT:
    targetProcess = batch_im2txt
    batch_im2txt.build_graph(config.VOCABFILE, config.IM2TXT_CHECKPOINT_DIR)
    outputDict[identifier + "_INIT"] = True

  # pooling
  while True:
    data = inputDict[identifier]
    if data is not None:
      log.info("new frame received")
      result = targetProcess.parse(data)
      del inputDict[identifier]
      inputDict[identifier] = None
      # send it back to queue
      outputDict[identifier] = result
    # else:
    #   # is this necessary?
    #   time.sleep(0.01)


if __name__ == "__main__":
  from types import SimpleNamespace
  config = SimpleNamespace()
  config.IS_PRODUCTION = False
  config.max_cal_width = "320"
  config.max_cal_height = "480"
  config.max_pred_width = 640
  config.max_pred_height = 720
  config.frame_skip = 2
  config.TF_MODEL_FOLDER = "/Users/eisneim/www/deepLearning/_tf_models"
  config.TF_MODEL_OD_CKPT_FOLDER = "/Volumes/raid/_deeplearning/_models/tf_detection_modelzoo/"
  config.COCO_LABEL_MAP_PATH = "data/mscoco_label_map_cn.pbtxt"

  config.VOCABFILE = "data/word_counts.txt"
  config.IM2TXT_CHECKPOINT_DIR = "/Users/eisneim/www/deepLearning/_pre_trained_model/im2txt/model.ckpt-2000000"

  # ----------- configure logging ---------
  log.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  # %(name)s -
  formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  log.addHandler(ch)
  # ---------------

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


