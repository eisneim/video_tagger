from flask import request, render_template, \
  jsonify, \
  g, \
  abort, \
  send_from_directory
from werkzeug.utils import secure_filename

import datetime
import logging

ALLOWED_EXTENSIONS = set(["mp4", "flv", "avi", "mov", "mkv", "webm", "ogv"])
log = logging.getLogger("vtr")

def allowed_file(filename):
  return "." in filename and \
    filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def routes(ctx):
  serverApp = ctx.app
  # mongo = ctx.mongo
  config = ctx.config
  tagger = ctx.tagger

  @serverApp.route("/")
  def index():
    header = {
      "X-Server": "Eisneim"
    }
    return render_template("index.html", name="people"), 200, header


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
      log.info("{}:{}".format(file.filename, file.content_type))
      # if user does not select file, browser also
      # submit a empty part without filename
      if file.filename == '':
          return jsonify({
            "err": "'video' form field is empty"
          })
      if file and allowed_file(file.filename):
        datestr = datetime.datetime.now().strftime("%m-%d_%H_%M%S")
        filename = datestr + "_" + secure_filename(file.filename)
        videopath = config.rootPath + "/uploads/" + filename
        file.save(videopath)

        err, state = tagger.parse(videopath)

        return jsonify(err=err, success=1, filename=filename,
          objectDetectionResults=state.objectDetectionResults,
          captioningResults=state.captioningResults,
          frameIndices=tagger.framesTobeParsedIdx)

    return jsonify({
        "err": "invalid filename"
      })
