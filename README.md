videoTagger
--------
video content analyzing and scene Tagging using computer vision and convolutional nerual networks.

## Key feature
 - object detection(yolo9000)
 - image captioning(im2txt)
 - action recongnition(c3d model)
 - face detection(facenet)
 - age detection

### dependencies
 - tensorflow 1.0
 - opencv3 python
 - numpy

### install thirdparty libs
```
cd path/To/videoTagger
mkdir thirdparty && cd thirdparty
```
#### install darkflow yolo9000
since officail darkflow doesn't support yolo9000 yet, so we need to use another cloned repo
```
git clone https://github.com/relh/darkflow
# follow the darkflow build instruction,
# note: we don't need to install it globally
```
download the [pretrained weights](http://pjreddie.com/media/files/yolo9000.weights) for yolo9000



