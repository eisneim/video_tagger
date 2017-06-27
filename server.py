import logging
from flask import Flask, request, render_template, \
  jsonify, \
  g, \
  abort, \
  send_from_directory
from werkzeug.utils import secure_filename
from flask_pymongo import PyMongo

from tagger import VideoTagger
import vconfig
from types import SimpleNamespace

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


# https://stackoverflow.com/questions/2827623/python-create-object-and-add-attributes-to-it
# create a ctx object to hold references
# ctx = SimpleNamespace()
# ctx.app = serverApp


# initialize our neural networks
# tagger = VideoTagger(config=config)

# set maxium file upload limit to 800Mb
# RequestEntityTooLarge exception will thrown for larger file
serverApp.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024
ALLOWED_EXTENSIONS = set(["mp4", "flv", "avi", "mov", "mkv", "webm", "ogv"])


def allowed_file(filename):
  return "." in filename and \
    filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# The following decorator registers a function
# on a list on the g object
def after_this_request(f):
  if not hasattr(g, 'after_request_callbacks'):
      g.after_request_callbacks = []
  g.after_request_callbacks.append(f)
  return f


@serverApp.after_request
def call_after_request_callbacks(response):
  for callback in getattr(g, 'after_request_callbacks', ()):
      callback(response)
  return response


@serverApp.before_request
def detect_user_language():
  language = request.cookies.get('user_lang')
  if language is None:
    language = "Unknown language" # guess_language_from_request()
    @after_this_request
    def remember_language(response):
      response.set_cookie('user_lang', language)
      print("deferred, set language in cookie")
  g.language = language
  print("language: {}".format(language))


@serverApp.errorhandler(404)
def page_not_found(error):
  return render_template('page_not_found.html'), 404


@serverApp.route("/")
def index():
  header = {
    "X-Server": "Eisneim"
  }
  return render_template("index.html", name="everyone"), 200, header

@serverApp.route("/json")
def jsonres():
  return jsonify({
      "status": 200,
      "info": "some info: {}".format(request.args)
    })


@serverApp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(serverApp.config['UPLOAD_FOLDER'],
                               filename)


@serverApp.route("/upload", methods=["POST"])
def upload():
  if request.method == "POST":
    # check if the post request has the file part
    if "video" not in request.files:
      return jsonify({
          "err": "no video uploaded, make sure the key of file form is 'video'"
        })
    file = request.files["video"]
    print("-----------file-------")
    print(dir(file))
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        return jsonify({
          "err": "'video' form field is empty"
        })
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(config.rootPath + "/uploads/" + filename)
      return jsonify(err=None, success=1)

  return jsonify({
      "err": "invalid filename"
    })


