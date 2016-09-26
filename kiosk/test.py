
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.layout import Layout
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.uix.behaviors import ButtonBehavior
from kivy.config import Config
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock

from kivy.uix.anchorlayout import AnchorLayout

from kivy.core.image import Image as CoreImage

import numpy as np
from scipy.interpolate import spline

import matplotlib.pyplot as plt
import seaborn as sns

import os
from os import listdir, mkdir
from os.path import isfile, join, exists

import time

import requests
import APIConstant

import time
# from Masking import doMasking


def getListByFloorAndLabel(floorId, label, maxDataPoints = -1):
    API_URL = APIConstant.API_HOST + APIConstant.LIST_BY_FLOOR_AND_LABEL + '/' + str(floorId) + '/' + label
    r = requests.get(API_URL);
    print 'getListByFloorAndLabel URL: ', API_URL

    if r.status_code != 200:
        print 'request: ', API_URL
        print 'Error, code: ', str(r.status_code)
        return None

    data = r.json()

    data = sorted(data, key=lambda x:x['created'])

    # for record in data:
    #     print record

    if maxDataPoints == -1 or maxDataPoints >= len(data):
        return data

    return data[len(data) - maxDataPoints :]

def getNodeListByFloor(floorId):
    API_URL = APIConstant.API_HOST + APIConstant.SEARCH_BY_FLOOR + '?' + 'floorId=' + str(floorId)
    r = requests.get(API_URL)
    print "api url: ", API_URL
    if r.status_code != 200:
        print 'Error, code ', str(r.status_code)
        return None

    nodes = r.json().get('items')

    # for node in nodes:
    #     print node

    return nodes

def drawChart(data, chartDirectory):

    size = len(data)
    if size == 0:
        print "drawChart: size == 0"

    y = [point['value'] for point in data]
    print 'chart value: ', y
    xtick = [point['marker'] for point in data]
    x = np.array(range(size))

    medium = [np.mean(y) for i in range(size)]

    xnew = np.linspace(x.min(), x.max(), size * 100)
    ynew = spline(x, y, xnew)
    ynew[ynew < 0] = 0

    plt.plot(xnew, ynew, x, medium)
    plt.fill_between(xnew, ynew, alpha=.3)

    plt.xticks(x, xtick)

    plt.xlabel('Time')
    plt.ylabel('Crowd Index')

    sns.set_style('dark')
    sns.despine(trim=True)

    # plt.show()

    if not exists(chartDirectory):
        mkdir(chartDirectory)

    filelist = [f for f in os.listdir(chartDirectory)]
    for f in filelist:
        os.remove(join(chartDirectory, f))

    _chartName = str(int(time.time())) + '.png'

    fig = plt.gcf()
    fig.set_size_inches(24, 13.5)
    fig.savefig(chartDirectory + _chartName)

    fig.clear()
    plt.close()
    return _chartName

def saveCameraPictures(nodes, imageDirectory):

    if not exists(imageDirectory):
        mkdir(imageDirectory)

    # filelist = [f for f in os.listdir(imageDirectory)]
    # for f in filelist:
    #     os.remove(join(imageDirectory, f))

    _imageList = []
    npics = 0

    for node in nodes:
        print "Node: ", node
        if not node.has_key('latestNodeFile') or node['latestNodeFile'] is None or not node['latestNodeFile'].has_key('fileUrl'):
            continue

        print node['latestNodeFile']['fileUrl']
        _image = node['latestNodeFile']['fileUrl'][node['latestNodeFile']['fileUrl'].rfind('/') + 1:]
        _imageList.append(_image)
        npics = npics + 1

        img_data = requests.get(node['latestNodeFile']['fileUrl']).content
        with open(imageDirectory + _image, 'wb') as handler:
            handler.write(img_data)

    print 'npics: ', npics
    print '_imageList: ', _imageList
    return _imageList

class ImageButton(ButtonBehavior, Image):
    pass

class MainScreen(GridLayout):

    def __init__(self, **kwargs):

        super(MainScreen, self).__init__(**kwargs)

        self.projectId = 1
        self.floorId = 11
        self.label = 'CrowdNow'
        self.maxDataPoints = 10

        self.chartDirectory = './chart/'
        self.imageDirectory = './pictures/'
        # self.maskedDirectory = './masked/'
        self.maskedDirectory = './pictures/'
        self.width = self.minimum_width

        # self.padding = 0
        # self.spacing = [10, 10]

        # with self.canvas.before:
        #     Color(1, 1, 1, 1)  # green; colors range from 0-1 instead of 0-255
        #     self.rect = Rectangle(size=self.size, pos=self.pos)
        #
        # self.bind(size=self._update_rect, pos=self._update_rect)

        self.imageList = []
        self.anchorList = []
        for i in range(4):
            if i % 2 == 0:
                self.anchorList.append(AnchorLayout(anchor_x='right', size=(800, 450), size_hint=(None, None)))
            else:
                self.anchorList.append(AnchorLayout(anchor_x='left', size=(800, 450), size_hint=(None, None)))

            self.imageList.append(ImageButton(source='./tmp_pics/1.jpg', size=(600, 400), on_press=self.showDialog))
            print 'anchor:', self.anchorList[i].size
            self.anchorList[i].add_widget(self.imageList[i])

            self.add_widget(self.anchorList[i])

        self.updateScreen()

    def updateScreen(self, *args):
        # floorData = getListByFloorAndLabel(self.floorId, self.label, self.maxDataPoints)
        # _chartName = drawChart(floorData, self.chartDirectory)

        # print 'length of floorData: ', len(floorData)
        # nodes = getNodeListByFloor(self.floorId)
        # _imageList = saveCameraPictures(nodes, self.imageDirectory)

        # _images = [join(self.imageDirectory, f) for f in _imageList if isfile(join(self.imageDirectory, f))]
        _images = ['./tmp_pics/1.jpg', \
                   './tmp_pics/2.jpg']
        # _chart = join(self.chartDirectory, _chartName)
        _chart = './chart/1474335850.png'

        # self.rows = 2
        # print self.rows

        self.imageIndex = 0
        for _image in _images:
            print '[DEBUG] source image: ', _image
            # self.image = ImageButton(title='Cameras',source=_image, on_press=self.showDialog)
            self.imageList[self.imageIndex].source = _image
            # print 'size after modified: ', self.imageList[self.imageIndex].size
            self.imageIndex = self.imageIndex + 1
            # self.add_widget(self.image)

        if len(_images) % 2 == 0:
            self.imageIndex = self.imageIndex + 1
        self.imageList[self.imageIndex].source = _chart

        print 'window size:', self.size
        print 'widget size:', self.imageList[0].size
        print 'width, minimum_width:', self.width, self.minimum_width

        # if os.path.isfile('./tmp_pics/1.jpg'):
        #     im = CoreImage('./tmp_pics/1.jpg')
        #     print 'Image size:', im.size
        #
        #     w_cols, w_rows = self.size
        #     x, y = im.size
        #     print w_cols, w_rows, x, y
        #     g = (w_cols / 2 - w_rows * x / y) + 10
        #     print 'g:', g
        #     self.spacing = [g, 0]
        # self.chart = ImageButton(title=_chartName,source=_chart,on_press=self.showDialog)
        # self.add_widget(self.chart)


    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def showDialog(self, instance):
        # print instance.source
        # dialog = Dialog(instance.source)
        # dialog.run()
        popup = Popup(title='bigger image', content=Image(source=instance.source), size_hint=(None, None), size=(1440, 810))
        popup.open()

class MainDisplay(App):

    def __init__(self):
        super(MainDisplay, self).__init__()

    def build(self):

        refreshGap = 10

        self.mainScreen = MainScreen()
        Clock.schedule_interval(self.mainScreen.updateScreen, refreshGap)
        return self.mainScreen

    def closeNow(self):
        self.get_running_app().stop()

if __name__ == '__main__':

    Config.set('graphics', 'width', '1600')
    Config.set('graphics', 'height', '900')
    # Config.set('graphics', 'fullscreen', 1)

    app = MainDisplay()
    app.run()


