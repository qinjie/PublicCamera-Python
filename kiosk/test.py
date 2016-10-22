import numpy as np
import datetime
from os.path import join, isfile
from os import listdir
from PIL import Image
from scipy.stats import mode as scimode


'''53s'''
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

    for i in range(len(colorImages)):
        colorImages[i] = np.reshape(colorImages[i], (1, colorImages[i].shape[0] * colorImages[i].shape[1], 3))[0]
        grayImages[i] = np.reshape(grayImages[i], (1, grayImages[i].shape[0] * grayImages[i].shape[1]))[0]

    background = np.zeros((rows * cols, 3), dtype='uint8')

    zipped = zip(*grayImages)

    for i, intensities in enumerate(zipped):
        mode = scimode(intensities)[0][0]
        idx = intensities.index(mode)
        background[i, :] = colorImages[idx][i, :]

    background = np.reshape(background, (rows, cols, 3))

    return background

'''9s'''
def getInterpolatedBackground2(PILImages):

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

    for i in range(len(colorImages)):
        colorImages[i] = np.reshape(colorImages[i], (1, colorImages[i].shape[0] * colorImages[i].shape[1], 3))[0]
        grayImages[i] = np.reshape(grayImages[i], (1, grayImages[i].shape[0] * grayImages[i].shape[1]))[0]

    zipped = zip(grayImages)
    zipped2 = zip(*grayImages)
    modes = scimode((zipped))[0][0][0]
    indexes = [zipped2[i].index(modes[i]) for i in range(rows * cols)]
    background = [colorImages[indexes[i]][i] for i in range(rows * cols)]
    background = np.reshape(background, (rows, cols, 3))

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

    begin_time = datetime.datetime.now()
    back = getInterpolatedBackground2(images)
    print 'processing time:', datetime.datetime.now() - begin_time
    back = Image.fromarray(back)
    back.show()