import cv2
import numpy as np
import os
import sys
import logging
import time
from multiprocessing import Process, Manager, current_process

from utils.util_img import ratio_scale_factor, save_thumbs
from utils.util_vis import drawBoxes, putCaptions
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
    self.scDetector = ContentDetector(
      threshold=int(config.THRESHOLD_SCENESEG),
      minFrames=int(config.SCENESEG_MINFRAMES))
    if config.PARALLEL:
      self.initialize_childProcess()
    else:
      self.framesTobeParsed = []
      self.framesTobeParsedIdx = []
      self.initialize_networks()

  def initialize_childProcess(self):
    qManager = Manager()
    self.inputDict = qManager.dict()
    self.outputDict = qManager.dict()
    # self.queue = Queue()

    # create neural network process
    self.process_od = Process(target=process_pooling,
      args=(self.inputDict, self.outputDict, PROCESS_OD, self.config),
      name=PROCESS_OD)
    self.process_od.daemon = True

    self.process_im2txt = Process(target=process_pooling,
      args=(self.inputDict, self.outputDict, PROCESS_IM2TXT, self.config),
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

  def initialize_networks(self):
    self.objectDetector = ObjectDetector(self.config,
      threshold=float(self.config.THRESHOLD_OBJECT_DETECTION))
    # build the network
    self.objectDetector.initialize()

    batch_im2txt.build_graph(self.config.VOCABFILE,
      self.config.IM2TXT_CHECKPOINT_DIR)

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
    self.videoName = os.path.basename(videoPath).rsplit(".")[0]
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
    donwScaleFactor = 1
    predDonwScaleFactor = 1
    if self.videoWidth > self.config.max_cal_width or \
      self.videoHeight > self.config.max_cal_height:
      scaleFactor = ratio_scale_factor(self.videoWidth,
        self.videoHeight,
        self.config.max_cal_width,
        self.config.max_cal_height)
      donwScaleFactor = round(1 / scaleFactor)
      log.info("donwScaleFactor: {}".format(donwScaleFactor))

    # donw-scale frame size for object detection
    if self.videoWidth > self.config.max_pred_width or \
      self.videoHeight > self.config.max_pred_height:
      scaleFactor = ratio_scale_factor(self.videoWidth,
        self.videoHeight,
        self.config.max_pred_width,
        self.config.max_pred_height)
      log.info("scaleFactor: {} {}".format(scaleFactor, round(1 / scaleFactor)))
      predDonwScaleFactor = round(1 / scaleFactor)
      log.info("predDonwScaleFactor: {}".format(donwScaleFactor))

    # store parsing state
    self.state = VideoParserState(donwScaleFactor,
      predDonwScaleFactor)

    # main loop
    self.loop(cap)
    # batchRecognize if the pipline is not running in parallel
    if not self.config.PARALLEL:
      od_results, caption_results = self.batchRecognize()
      self.state.objectDetectionResults = od_results
      self.state.captioningResults = caption_results

    # Cleanup and return parse state.
    cap.release()
    return None, self.state

  def loop(self, cap):
    skip = 0
    if self.config.frame_skip and int(self.config.frame_skip) > 0:
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
      log.info("cut found at: {}".format(self.state.frameNum))

  def sceneDetectedParallel(self, frames):
    if not self.allChildProcessReady:
      self.blockUntilChildprocessReady()

    # boxes, scores, classes, num_detections
    # detectedResults = self.objDetector.parse(
    #   [dominateFrame])
    self.inputDict[PROCESS_OD] = frames
    self.inputDict[PROCESS_IM2TXT] = frames
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
    return resultData

  def sceneDetected(self, isLastFrame=False):
    if not isLastFrame:
      self.state.scenes.append(self.state.frameNum)
    # should find best frame to represent this scene
    # currently using the dumbest way, which is using the middle frame
    # @TODO: better way to find representative frame
    midIndex = len(self.state.currentSceneFrames) // 2
    frameNumber = self.state.frameNum + midIndex
    dominateFrame = self.state.currentSceneFrames[midIndex]
    # empty scenFrames buffer
    self.state.currentSceneFrames = []
    frames = [dominateFrame]
    # >>> should start object detection on dominateFrame
    if self.config.PARALLEL:
      resultData = self.sceneDetectedParallel(frames)
      # visualize boundingboxes
      if not config.IS_PRODUCTION:
        for result in resultData[PROCESS_OD]:
          boxes, scores, classes, _ = result
          img = drawBoxes(dominateFrame.copy(), boxes, scores, classes)

        cv2.imshow("detection", img)
        cv2.waitKey(5)
    else:
      self.framesTobeParsed.append(dominateFrame)
      self.framesTobeParsedIdx.append(frameNumber)
    # should save the frame(s) to filesystem

  def batchRecognize(self):
    log.info("frames to be recognized: {}".format(len(self.framesTobeParsed)))
    od_results = self.objectDetector.parse(self.framesTobeParsed)
    caption_results = batch_im2txt.parse(self.framesTobeParsed)

    log.info("len(od results): {}".format(len(od_results)))
    log.info("len(caption results): {}".format(len(caption_results)))

    imgsToSave = self.framesTobeParsed
    if not self.config.IS_PRODUCTION:
      imgsToSave = []
      for idx, frame in enumerate(self.framesTobeParsed):
        captions = caption_results[idx]
        boxes, scores, classes, _  = od_results[idx]
        img = putCaptions(frame.copy(), captions)
        img = drawBoxes(img, boxes, scores, classes)
        # cv2.imshow("results", img)
        # cv2.waitKey(300)
        imgsToSave.append(img)
    # save images to file
    save_thumbs(self.videoName, imgsToSave,
      self.framesTobeParsedIdx,
      self.config.THUMB_DIR)

    return od_results, caption_results

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
  config.PARALLEL = False
  config.IS_PRODUCTION = False
  config.max_cal_width = "320"
  config.max_cal_height = "480"
  config.max_pred_width = 640
  config.max_pred_height = 720
  config.frame_skip = 2
  config.THUMB_DIR = "thumbs"
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
  print("scenes: {}".format(state.scenes))
  print("self.sceneList: ", tagger.sceneList)


