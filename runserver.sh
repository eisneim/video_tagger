#!/usr/bin/env bash

# this is used for importing tensorflow models
# more instruction here: https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md
# modelPath=/Users/eisneim/www/deepLearning/_tf_models
# export PYTHONPATH=$PYTHONPATH:${modelPath}:${modelPath}/slim

# activates the debugger, activates the automatic reloader
export FLASK_DEBUG=1

export FLASK_APP=server.py
flask run --host=0.0.0.0
