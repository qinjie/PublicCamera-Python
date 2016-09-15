import ConfigParser
import calendar
import json
import os
import sys
import threading
from datetime import date, datetime
from time import sleep

import picamera
from requests.auth import HTTPBasicAuth

from floorsetting import FloorSetting
from node import Node
from nodefile import NodeFile
from rpi_utils import get_logger
from utils import time_in_range, make_sure_path_exist

__author__ = 'zqi2'

if __name__ == '__main__':

    # # Read options from command line
    # argParser = argparse.ArgumentParser('API Entity')
    # argParser.add_argument('-c', '--configFile', help="Configuration file", required=False)
    # argParser.add_argument('-s', '--configSession', help="Configuration session", required=False)
    # argParser.add_argument('-u', '--username', help="Username", required=False)
    # argParser.add_argument('-p', '--password', help="Password", required=False)
    # args = argParser.parse_args()


    # configuration file and section name
    _config_file = 'batch.ini'
    # use <country> for testing purpose
    _config_section = 'default'

    parser = ConfigParser.SafeConfigParser()
    parser.read(_config_file)
    parser.defaults()

    FLOOR_ID = parser.getint('default', 'floor_id')
    NODE_ID = parser.getint('default', 'node_id')
    username = parser.get('default', 'username')
    password = parser.get('default', 'password')
    auth = HTTPBasicAuth(username, password)
    setting_interval = parser.getint('default', 'interval_seconds')
    PHOTO_PATH = parser.get('default', 'photo_path')
    batch_duration_seconds = parser.getint('default', 'batch_duration_seconds')

    # Get the logger
    logger = get_logger(name='batch_take_photos', reset_handlers=True, log_file='batch.log', log_console=True)

    # Get node label
    node = Node()
    nr = node.view(NODE_ID)
    nj = json.loads(nr.text)
    node_label = nj['label']

    # Get floor settings form web service
    entity = FloorSetting()
    resp = entity.search('floorId={0}'.format(FLOOR_ID), auth)
    j = json.loads(resp.text)
    if 'items' not in j:
        logger.error("No items in returned json")
        sys.exit()

    # Create a setting map for this floor
    setting_map = {}
    for obj in j['items']:
        setting_map[obj['label']] = obj['value']
    logger.info(setting_map)

    # check weekday
    logger.info("Check weekday setting")
    my_date = date.today()
    my_weekday = calendar.day_name[my_date.weekday()].lower()
    if my_weekday in setting_map:
        if int(setting_map[my_weekday]) == 0:
            logger.info("No work on " + my_weekday)
            sys.exit()

    # Check start time and end time
    if ('start_time' in setting_map) and ('end_time' in setting_map):
        print "Check start_time and end_time settings"
        start_time = datetime.strptime(setting_map['start_time'], '%H:%M').time()
        end_time = datetime.strptime(setting_map['end_time'], '%H:%M').time()
        cur_time = datetime.now().time()

        if not time_in_range(start_time, end_time, cur_time):
            logger.info("No work outside {0} and {1}".format(start_time, end_time))
            sys.exit()

    # Update interval
    if 'interval' in setting_map:
        setting_interval = int(setting_map['interval'])
        logger.info("Interval setting from webservice = {0}".format(setting_interval))


    def upload_photo(args):
        out_file = str(args)
        if not os.path.isfile(out_file):
            logger.error("file not found: {0}".format(out_file))
            return
        entity = NodeFile()

        # UPLOAD
        payload = {'nodeId': NODE_ID, 'fileName': out_file, 'label': node_label}
        r = entity.upload(payload, auth)
        if r.status_code == 200:
            logger.info("Upload successful: {0}".format(out_file))


    def threaded_photo_upload(my_arg):
        thread = threading.Thread(target=upload_photo, args=my_arg)
        thread.start()


    def init_camera(camera):
        camera.vflip = False
        camera.hflip = False
        camera.brightness = 50
        camera.resolution = (1920, 1080)
        camera.framerate = 24
        # camera.annotate_background = picamera.Color('black')
        camera.annotate_text_size = 12
        camera.annotate_foreground = picamera.Color('white')


    def take_photo(i):
        with picamera.PiCamera() as camera:
            init_camera(camera)

            #camera.rotation = 90
            t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            camera.annotate_text = "{0} {1}".format(node_label, t)
            out_file = os.path.join(PHOTO_PATH, 'image%03d.jpg' % i)

            camera.start_preview()
            sleep(0.5)
            camera.capture(out_file)
            sleep(0.5)
            camera.stop_preview()

            threaded_photo_upload([out_file])


    def take_photos():
        count = batch_duration_seconds // setting_interval

        for i in range(count):
            threading.Timer(i * setting_interval, take_photo, (i,)).start()


    # take photos
    make_sure_path_exist(PHOTO_PATH)
    take_photos()

