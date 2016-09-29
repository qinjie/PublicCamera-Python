import numpy as np
import statistics
from os.path import join, isfile
from os import listdir
from PIL import Image
from scipy.stats import mode as scimode

def getInterpolatedBackground(PILImages):

    colorImages = []
    grayImages = []

    for PILImage in PILImages:
        colorImage = np.array(PILImage)
        grayImage = np.array(colorImage[:, :, 0] * 0.299 + colorImage[:, :, 1] * 0.587 + colorImage[:, :, 2] * 0.114).astype('uint8')

        colorImages.append(colorImage)
        grayImages.append(grayImage)

    for image in grayImages:
        if not np.array_equal(grayImages[0].shape, image.shape):
            print 'Image shapes are different'
            return None

    rows, cols = grayImages[0].shape

    background = np.zeros((rows, cols, 3), dtype='uint8')

    for i in range(rows):
        for j in range(cols):
            intensities = [grayImages[k][i, j] for k in range(0, len(grayImages))]
            mode = scimode(intensities)[0][0]
            idx = intensities.index(mode)

            background[i, j, :] = colorImages[idx][i, j, :]

    return background


if __name__ == '__main__':
    _path = './pictures/'

    _images = [join(_path, f) for f in listdir(_path) if isfile(join(_path, f))]

    images = []
    for _image in _images:
        try:
            image = Image.open(_image)
            images.append(image)
        except Exception:
            pass

    back = getInterpolatedBackground(images)
    back = Image.fromarray(back)
    back.show()