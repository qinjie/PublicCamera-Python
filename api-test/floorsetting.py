__author__ = 'zqi2'

import ConfigParser
import argparse
from requests.auth import HTTPBasicAuth
from entity import Entity

class FloorSetting (Entity):
    # config section
    _config_section = 'floor_setting'


if __name__ == '__main__':

    # # Read options from command line
    # argParser = argparse.ArgumentParser('Public Camera: API Entity')
    # argParser.add_argument('-c', '--configFile', help="Configuration file", required=False)
    # argParser.add_argument('-s', '--configSession', help="Configuration session", required=False)
    # argParser.add_argument('-u', '--username', help="Username", required=False)
    # argParser.add_argument('-p', '--password', help="Password", required=False)
    # args = argParser.parse_args()

    # Username and Password for Authentication
    username = 'user1'
    password = '123456'
    auth = HTTPBasicAuth(username, password)

    entity = FloorSetting()

    # LIST
    entity.list()

    # VIEW
    entity.view(1)

    # SEARCH
    entity.search('label=interval', auth)

    # CREATE
    data = {'floorId': '1', 'label': 'test', 'value': '123'}
    r = entity.create(data, auth)

    if r.status_code == 201:

        # UPDATE
        obj = r.json()
        obj['label'] = 'test123'
        obj['value'] = '345'
        r2 = entity.update(obj, auth)

        # DELETE
        r3 = entity.delete(obj['id'], auth)