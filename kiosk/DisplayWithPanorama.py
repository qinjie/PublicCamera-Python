'''
The title is hard-coded here
The panorama is taken from server
Chart is drawed from data taken from server
'''

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.stacklayout import StackLayout
from kivy.uix.image import Image
from kivy.config import Config
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.uix.label import Label
from PIL import Image as PilImage


import matplotlib
matplotlib.use('Agg')

import numpy as np
from scipy.interpolate import spline

import matplotlib.pyplot as plt
import seaborn as sns

import os
from os import listdir, mkdir
from os.path import isfile, join, exists

import requests
from requests.exceptions import ConnectionError
import APIConstant

import time


def getListByFloorAndLabel(floorId, label, maxDataPoints = -1):
    API_URL = APIConstant.API_HOST + APIConstant.LIST_BY_FLOOR_AND_LABEL + '/' + str(floorId) + '/' + label
    r = requests.get(API_URL);
    # print 'getListByFloorAndLabel URL: ', API_URL

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
    # print "api url: ", API_URL
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
    # print 'chart value: ', y
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
    fig.savefig(chartDirectory + _chartName, bbox_inches='tight')

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
        # print "Node: ", node
        if not node.has_key('latestNodeFile') or node['latestNodeFile'] is None or not node['latestNodeFile'].has_key('fileUrl'):
            continue

        # print node['latestNodeFile']['fileUrl']
        _image = node['latestNodeFile']['fileUrl'][node['latestNodeFile']['fileUrl'].rfind('/') + 1:]
        _imageList.append(_image)
        npics = npics + 1

        img_data = requests.get(node['latestNodeFile']['fileUrl']).content
        with open(imageDirectory + _image, 'wb') as handler:
            handler.write(img_data)

        basewidth = 960
        img = PilImage.open(imageDirectory + _image)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), PilImage.ANTIALIAS)
        img.save(imageDirectory + _image)

    # print 'npics: ', npics
    # print '_imageList: ', _imageList
    return _imageList

class MainScreen(StackLayout):

    def __init__(self, **kwargs):

        super(MainScreen, self).__init__(**kwargs)

        self.projectId = 1
        self.floorId = 11
        self.label = 'CrowdNow'
        self.maxDataPoints = 10

        self.chartDirectory = './chart/'
        self.imageDirectory = './pictures/'
        self.maskedDirectory = './masked/'
        self.maskedDirectory = './pictures/'

        '''The below values must be summed up to 1'''
        self.title_height_p = 0.1
        self.pano_height_p = 0.45
        self.chart_height_p = 0.45

        self.pano = None

        with self.canvas.before:
            Color(1, 1, 1, 1)  # colors range from 0-1 instead of 0-255
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)

        self.initScreen()

    def initScreen(self, img_size=None):

        print 'init screen'

        self.clear_widgets()
        self.titleLabel = Label(text='[color=654321][size=30]Lab room[/size][/color]',
                                markup=True, size=[self.size[0], int(self.size[1] * self.title_height_p)],
                                size_hint=(None, None))
        self.titleLayout = GridLayout(rows=1, size=self.titleLabel.size, size_hint=(None, None))
        self.titleLayout.add_widget(self.titleLabel)
        self.add_widget(self.titleLayout)

        self.last_size = np.copy(self.size)

        img_row = int(self.size[1] * self.pano_height_p)
        img_col = self.size[0]

        '''Setup the pano layout'''
        self.pano = Image(size=(img_col, img_row), size_hint=(None, None))
        self.panoLayout = GridLayout(rows=1, size=self.pano.size, size_hint=(None, None))
        self.panoLayout.add_widget(self.pano)
        self.add_widget(self.panoLayout)

        chart_row = int(self.size[1] * self.chart_height_p)
        chart_col = int(self.size[0] / 2)

        self.chart = Image(size=(chart_col, chart_row), size_hint=(None, None))
        self.chartLayout = GridLayout(rows=1, size=(self.size[0],int(self.size[1] * self.chart_height_p)),
                                      size_hint=(None, None))
        self.chartLayout.add_widget(Widget()) # add a blank widget
        self.chartLayout.add_widget(self.chart)
        self.add_widget(self.chartLayout)

        self.updateScreen()

    def updateScreen(self, *args):

        try:
            floorData = getListByFloorAndLabel(self.floorId, self.label, self.maxDataPoints)

            _chartName = drawChart(floorData, self.chartDirectory)

            nodes = getNodeListByFloor(self.floorId)

            #TODO: the _pano must be changed to the link to the panorama taken from server
            _pano = './panoAB.jpg'

            _chart = join(self.chartDirectory, _chartName)
        except ConnectionError:
            print 'just got ConnectionError'
            return

        self.pano.source = _pano
        self.chart.source = _chart

        # print 'size and last_size', self.size, self.last_size
        if not np.array_equal(self.size, self.last_size):
            # print 'window size changed'
            image = Image(source=_pano)
            self.initScreen(image.texture.size)


    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

class MainDisplay(App):

    def __init__(self):
        super(MainDisplay, self).__init__()

    def build(self):

        refreshGap = 20

        self.mainScreen = MainScreen()
        Clock.schedule_interval(self.mainScreen.updateScreen, refreshGap)
        return self.mainScreen


if __name__ == '__main__':

    Config.set('graphics', 'width', '1600')
    Config.set('graphics', 'height', '900')

    sns.set(font_scale=3)

    app = MainDisplay()
    app.run()
