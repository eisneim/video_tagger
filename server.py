import logging
from flask import Flask, request

from tagger import VideoTagger
import vconfig

# ----------- configure logging ---------
log = logging.getLogger("vtr")
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# %(name)s -
formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
ch.setFormatter(formatter)

log.addHandler(ch)
# ------  end of configure logging ---------
# initialize the server
serverApp = Flask(__name__)

config = vconfig.VConfig()
# initialize our neural networks
tagger = VideoTagger(config=config)


@serverApp.route("/")
def index():
  return "index page"


@serverApp.route("/upload", methods=["POST"])
def upload():
  if request.method = "POST":
    print("should deal with post file")
  return "this is the result"

