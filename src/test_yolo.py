import urllib.request

url = "https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_swin-t_fpn_3x_det_bdd100k.pth"

urllib.request.urlretrieve(url, "swin_bdd100k.pth")