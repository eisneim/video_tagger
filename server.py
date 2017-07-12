import logging
from flask import Flask
from flask_pymongo import PyMongo

from tagger import VideoTagger
import vconfig
from types import SimpleNamespace

from routeHandlers import routes

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
log.info("root path: {}".format(config.rootPath))
serverApp.config["UPLOAD_FOLDER"] = config.rootPath + "/uploads/"

# --- setup mongodb ------
serverApp.config["MONGO_DBNAME"] = "videoTagger"
serverApp.config["MONGO_URI"] = "mongodb://localhost:27017/videoTagger"
mongo = PyMongo(serverApp)

# set maxium file upload limit to 800Mb
# RequestEntityTooLarge exception will thrown for larger file
serverApp.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024

# initialize our neural networks
tagger = VideoTagger(config=config)

# https://stackoverflow.com/questions/2827623/python-create-object-and-add-attributes-to-it
# create a ctx object to hold references
ctx = SimpleNamespace()
ctx.app = serverApp
ctx.mongo = mongo
ctx.config = config
ctx.tagger = tagger

routes(ctx)

