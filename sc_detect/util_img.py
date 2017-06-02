def ratio_scale_factor(width, height, maxw, maxh):
  zoomfactor = 1
  ratio = width / height
  setedRatio = maxw / maxh
  # if image size is too big, resize it to fit window size and display
  destw = desth = 0
  if ratio >= setedRatio and width > maxw:
    destw = maxw
    desth = maxw / ratio
    # print("landscape: {}x{} => {}x{}".format(width, height, destw, desth))
  elif ratio < setedRatio and height > maxh:
    desth = maxh
    destw = maxh * ratio
    # print("portrait: {}x{} => {}x{}".format(width, height, destw, desth))
  else:
    # no resize required
    zoomfactor
  zoomfactor = destw / width
  return zoomfactor