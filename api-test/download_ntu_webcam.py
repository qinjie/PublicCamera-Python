import datetime
import os

__author__ = 'zqi2'

import requests
import shutil


def download_image(url, path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)


images = {
    'fastfood': 'http://webcam.ntu.edu.sg/upload/slider/fastfood.jpg',
    'foodcourt': 'http://webcam.ntu.edu.sg/upload/slider/foodcourt.jpg',
    'lwninside': 'http://webcam.ntu.edu.sg/upload/slider/lwn-inside.jpg',
    'quad': 'http://webcam.ntu.edu.sg/upload/slider/quad.jpg',
    'onestop': 'http://webcam.ntu.edu.sg/upload/slider/onestop_sac.jpg',
    'canteenB': 'http://webcam.ntu.edu.sg/upload/slider/canteenB.jpg',
    'Walkway': 'http://webcam.ntu.edu.sg/upload/slider/WalkwaybetweenNorthAndSouthSpines.jpg'
}

folder = 'testfiles'
if not os.path.isdir(folder):
     os.makedirs(folder)

i = datetime.datetime.now()
dt = i.strftime('%Y%m%d-%H%M%S')
print dt

for k, v in images.iteritems():
    path = folder + '/' + dt + '-' + k + '.jpg'
    # path = dt + '-' + k + '.jpg'
    print 'saving ' + path
    download_image(v, path)
