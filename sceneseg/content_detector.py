import cv2
import numpy as np
from sceneseg import SceneSeg


class ContentDetector(SceneSeg):
  def __init__(self, threshold=30, minFrames=15):
    super(ContentDetector, self).__init__()

    self.threshold = threshold
    self.minFrames = minFrames
    self.lastFrame = None
    self.lastSceneFrameNum = None
    self.lastHSV = None
    self.lastSceneCut = None

  def process_frame(self, frameNum, frameImg, frameMetrics, sceneList):

    cutDetected = False

    if self.lastFrame is None:
      self.lastFrame = frameImg.copy()
      return cutDetected

    # change in average of HSV (hsv), (h)ue only, (s)aturation only, (l)uminance only.
    delta_hsv_avg, delta_h, delta_s, delta_v = 0.0, 0.0, 0.0, 0.0

    if frameNum in frameMetrics and 'delta_hsv_avg' in frameMetrics[frameNum]:
      delta_hsv_avg = frameMetrics[frameNum]['delta_hsv_avg']
      delta_h = frameMetrics[frameNum]['delta_hue']
      delta_s = frameMetrics[frameNum]['delta_sat']
      delta_v = frameMetrics[frameNum]['delta_lum']

    else:
      numPixels = frameImg.shape[0] * frameImg.shape[1]
      currHSV = cv2.split(cv2.cvtColor(frameImg, cv2.COLOR_BGR2HSV))
      lastHSV = self.lastHSV

      if not lastHSV:
        lastHSV = cv2.split(cv2.cvtColor(self.lastFrame, cv2.COLOR_BGR2HSV))

      deltaHSV = [-1, -1, -1]
      # for each channel, calcl differience
      for ii in range(3):
        currHSV[ii] = currHSV[ii].astype(np.int32)
        lastHSV[ii] = lastHSV[ii].astype(np.int32)
        deltaHSV[ii] = np.sum(np.abs(currHSV[ii] - lastHSV[ii])) / float(numPixels)
      deltaHSV.append(sum(deltaHSV) / 3.0)
      delta_h, delta_s, delta_v, delta_hsv_avg = deltaHSV

      frameMetrics[frameNum]['delta_hsv_avg'] = delta_hsv_avg
      frameMetrics[frameNum]['delta_hue'] = delta_h
      frameMetrics[frameNum]['delta_sat'] = delta_s
      frameMetrics[frameNum]['delta_lum'] = delta_v

      self.lastHSV = currHSV

      if delta_hsv_avg >= self.threshold:
        if self.lastSceneCut is None or (
          (frameNum - self.lastSceneCut) >= self.minFrames):
          sceneList.append(frameNum)
          self.lastSceneCut = frameNum
          cutDetected = True
      del self.lastFrame

    self.lastFrame = frameImg.copy()
    return cutDetected

  def post_process(self, scene_list):
    return

