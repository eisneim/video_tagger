import requests
import logging
import requests
import base64
# import json
import librosa


log = logging.getLogger("vtr")

URL_TOKEN = "https://openapi.baidu.com/oauth/2.0/token?\
grant_type=client_credentials&client_id={}&\
client_secret={}&"
URL_TARGET = "http://vop.baidu.com/server_api"


class Baiduasr:
  def __init__(self, cuid, appid, appsecret,
    token=None, channel=1, rate=1600, fileformat="wav"):
    self.cuid = cuid
    self.appid = appid
    self.appsecret = appsecret
    self.token = token
    self.channel = channel
    self.rate = rate
    self.format = fileformat


  def get_access_token(self):
    """
    {
    6. "access_token": "1.a6b7dbd428f731035f771b8d*******",
    7. "expires_in": 86400,
    8. "refresh_token": "2.385d55f8615fdfd9edb7c4b******",
    9. "scope": "public",
    10. "session_key": "ANXxSNjwQDugf8615Onqeik********CdlLx n",
    11. "session_secret": "248APxvxjCZ0VEC********aK4oZExMB ",
    12. }
    """
    url = URL_TOKEN.format(self.appid, self.appsecret)
    log.info("request: {}".format(url))
    r = requests.get(url)
    self.tokenRes = r.json()
    if "error_description" in self.tokenRes:
      return (self.tokenRes["error_description"], False)

    self.token = self.tokenRes["access_token"]
    return (None, self.tokenRes)

  def speech_recon(self, audiopath,
    rate=None, format=None, channel=None):
    # get proper sample rate:
    if not rate:
      _, rate = librosa.load(audiopath, sr=None)
      print("sample rate:", rate)

    payload = {
      "token": self.token,
      "cuid": self.cuid,
      "format": format or self.format,
      "rate": rate or self.rate,
      "channel": channel or self.channel,
    }

    with open(audiopath, "rb") as fin:
      audio = fin.read()
      audiostr = base64.b64encode(audio).decode()
      print("base64 length", len(audiostr))

    payload["sppech"] = audiostr
    payload["len"] = len(audio)

    try:
      rr = requests.post(URL_TARGET, data=payload, stream=True)
    except requests.exceptions.RequestException as e:
      msg = "Connection timeout please retry"
      return (msg, None)

    return self.parse_res(rr)

  def parse_res(self, rr):
    ctype = rr.headers["content-type"]
    isJson = ctype.find("json") >= 0
    jsonRes = { "err_no": None, "err_msg": "unknow Error" }
    if isJson:
      jsonRes = rr.json()
    if rr.status_code == 500 or jsonRes["err_no"] ==500:
      msg = "不支持输入"
      log.error(msg)
      return (msg, None)
    elif rr.status_code == 501 or jsonRes["err_no"] ==501:
      msg = "输入参数不正确"
      log.error(msg)
      return (msg, None)
    elif rr.status_code == 502 or jsonRes["err_no"] ==502:
      msg = "token 验证失败"
      log.error(msg)
      return (msg, None)
    elif rr.status_code == 503 or jsonRes["err_no"] ==503:
      msg = "合成后端错误"
      log.error(msg)
      return (msg, None)
    elif rr.status_code == 404 or jsonRes["err_no"] ==404:
      msg = "合成后端地址错误"
      log.error(msg)
      return (msg, None)

    log.info('response type: {} {}'.format(ctype, rr.status_code))
    if rr.status_code == 200:
      log.info("deal with raw audio file")
      return (None, rr)
    else:
      log.info("show response error: {}".format(rr.text))
      return (rr.text, None)

  def speech_recon2(self, audiopath, sr=None, format="wav"):
    # get proper sample rate:
    if not sr:
      _, sr = librosa.load(audiopath, sr=None)
      print("sample rate:", sr)

    with open(audiopath, "rb") as fin:
      audio = fin.read()

    files = { "audio": audio }
    headers = {
      "Content-Type": "audio/{}; rate={}".format(format, sr)
    }
    url = URL_TARGET + "?cuid={}&token={}".format(self.cuid, self.token)
    rr = requests.post(url, files=files, headers=headers)
    return self.parse_res(rr)


if __name__ == "__main__":
  token="24.a5a0747dfc5d229a0da4eb4ca04fa055.2592000.1499423869.282335-5013978"
  api = Baiduasr("5013978",
    "oDjb5r5XVpW04Gopd374muGK",
    "53A9ufobHzijbaRhI5QTidXUu5v788rO", token=token)
  # err, res = api.get_access_token()
  # print("res:", res)
  # if err:
  #   raise err

  # test request
  audiopath = "/Users/eisneim/www/work/videoTagger/testData/audio/dialog1.wav"
  err, res = api.speech_recon2(audiopath)
  if err:
    raise err

  print(res.text)



