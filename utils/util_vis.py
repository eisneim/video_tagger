import cv2
import numpy as np


def drawBoxes(img, boxes, scores, names):
  color = (0, 255, 0)
  font = cv2.FONT_HERSHEY_SIMPLEX
  thickness = 1
  height, width, _ = img.shape
  for idx, box in enumerate(boxes):
    ymin, xmin, ymax, xmax = box
    ymin = int(ymin * height)
    ymax = int(ymax * height)
    xmin = int(xmin * width)
    xmax = int(xmax * width)
    print("points: {} {}".format((xmin, ymin), (xmax, ymax)))
    name = names[idx]
    score = scores[idx]
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    # draw classname
    cv2.putText(img,
      "{}: {:.3f}".format(name, score),
      (xmin, ymin - 10),
      font, 0.5,
      color,
      thickness)

  return img


def putCaptions(img, captions):
  color = (255, 0, 0)
  font = cv2.FONT_HERSHEY_SIMPLEX
  thickness = 1
  txtHeight = 24
  for idx, cap in enumerate(captions):
    sentence = cap["sentence"]
    p = cap["p"]
    cv2.putText(img,
      "({}) {} ({:.5f})".format(idx, " ".join(sentence), p),
      (5, txtHeight * idx + 15),
      font, 0.5,
      color,
      thickness)

  return img