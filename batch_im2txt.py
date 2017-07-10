from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import sys

import tensorflow as tf
import cv2

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

tf.logging.set_verbosity(tf.logging.INFO)

# build the inference graph
gg = tf.Graph()
vocab = None
restore_fn = None
model = None
modelConfig = None


def build_graph(vocab_file, checkpoint_path):
  """initalize tensorflow graph

  Arguments:
    vocab_file {str} -- vocabulary txt file location
    checkpoint_path {str} -- folder path of checkpoint files
  """
  global vocab
  global gg
  global model
  global restore_fn
  global modelConfig

  with gg.as_default():
    model = inference_wrapper.InferenceWrapper()
    modelConfig = configuration.ModelConfig()
    # use numpy image as input so we don't need to decode jpg file
    modelConfig.skip_decode = True

    restore_fn = model.build_graph_from_config(modelConfig,
                                               checkpoint_path)
  gg.finalize()
  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(vocab_file)


def parse(images):
  """batch generate image captions

  Arguments:
    images {list} -- list of numpy image
  """
  tf.logging.info("Running caption generation on %d images",
                  len(images))

  with tf.Session(graph=gg) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)
    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See im2txt/caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    allCaptions = []

    for img in images:
      img = cv2.resize(img, (modelConfig.image_width, modelConfig.image_height))

      captions = generator.beam_search(sess, img)
      allCaptions.append(captions)

      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
      print("============")
    return allCaptions


