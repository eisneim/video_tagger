import cv2
import numpy as np
import sys


class FaceDetector(object):
  def __init__(self):
    pass

  def run(self, img):
    pass


class FaceDetectorCascadeOpenCV(FaceDetector):

  def __init__(self, frontalModel='data/haarcascade_frontalface_default.xml',
    profileModel="data/haarcascade_profileface.xml"):
    self.faceCascade = cv2.CascadeClassifier(frontalModel)
    self.profileCascade = cv2.CascadeClassifier(profileModel)

  def run(self, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = self.faceCascade.detectMultiScale(gray, 1.3, minNeighbors=5)
    facesImg = []
    for (x, y, w, h) in faces:
      roiColor = img[y: y + h, x: x + w]
      facesImg.append(roiColor)

    return faces, facesImg

  def visualize(self, img, faces):
    for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("facedetection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def testOcv():
  detector = FaceDetectorCascadeOpenCV()
  print("image: {}".format(sys.argv[1]))
  img = cv2.imread(sys.argv[1])
  faces, facesImg = detector.run(img)
  detector.visualize(img, faces)

if __name__ == "__main__":
  testOcv()


