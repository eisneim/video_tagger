import cv2
import numpy as np
import os

import scene_detect


def main():
  video = "testData/video.mp4"
  rootPath = os.path.dirname(os.path.realpath(__file__))
  thumb_path = os.path.join(rootPath, "thumbs/")
  if not os.path.exists(thumb_path):
    os.mkdir(thumb_path)

  fps, frameCount, scenes = scene_detect.detect(video,
    thumb_path=thumb_path)

  print(fps, frameCount)
  print(scenes)


if __name__ == "__main__":
  main()