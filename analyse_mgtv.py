
import pymongo
from types import SimpleNamespace
from tagger import VideoTagger
import vconfig


config = vconfig.VConfig()
COL_DRAMA = "dramas"
COL_COMMENT = "comments"
COL_VIDEO = "vidresults"


def analyse_video(ctx, video, drama):
  tagger = VideoTagger(config=config)
  err, state, frameParseResults = tagger.parse(video["videofile"])
  if err:
    print("{}/{}".format(drama["title"], video["t4"]), err)
  else:
    # save results
    ctx.db[COL_VIDEO].insert({
      "video_id": video["video_id"],
      "drama_id": drama["id"],
      "analyse_results": frameParseResults,
      "scenes": state.scenes,
      "total_frames": state.frameNum,
      "fps": state.fps,
      "width": state.width,
      "height": state.height,
    })



def main():
  ctx = SimpleNamespace()
  client = pymongo.MongoClient(config.MONGO_HOST, config.MONGO_PORT)
  ctx.db = client[config.MONGO_DBNAME]
  dramas = ctx.db[COL_DRAMA].find({}).select({
    "episodes": 1, "title": 1, "id": 1,
  })

  for drama in dramas:
    print(">>> analyse drama: {}".format(drama["title"]))
    for video in drama["episodes"]:
      if "videofile" not in video:
        print("skip: {}:{}, no video file".format(drama["title"], video["t4"]))
        continue
      analyse_video(ctx, video, drama)
