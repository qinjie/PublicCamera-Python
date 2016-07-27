__author__ = 'zqi2'

import argparse
import ConfigParser
import requests
from requests.auth import HTTPBasicAuth
from entity import Entity

class FloorData (Entity):
    # config section
    _config_section = 'floor_data'

    def __init__(self):
        Entity.__init__(self)

        parser = ConfigParser.SafeConfigParser()
        parser.read(self._config_file)
        self._urls['latest_by_project'] = self._base_url + parser.get(self._config_section, 'url_latest_by_project')
        self._urls['latest_by_project_and_label'] = self._base_url + parser.get(self._config_section, 'url_latest_by_project_and_label')

    def latest_by_project(self, projectId, auth=None):

        url = self._urls['latest_by_project'].replace("<projectId>", str(projectId))
        headers = {'Accept': 'application/json'}
        r = requests.get(url, auth=auth, headers=headers)
        self.log.info("LATEST_BY_PROJECT: %s", url)
        self.log.info("%s %s", r.status_code, r.headers['content-type'])
        self.log.info(r.text)
        return r

    def latest_by_project_and_label(self, projectId, label, auth=None):

        url = self._urls['latest_by_project_and_label'].replace("<projectId>", str(projectId))
        url = url.replace("<label>", str(label))
        headers = {'Accept': 'application/json'}
        r = requests.get(url, auth=auth, headers=headers)
        self.log.info("LATEST_BY_PROJECT: %s", url)
        self.log.info("%s %s", r.status_code, r.headers['content-type'])
        self.log.info(r.text)
        return r


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

    entity = FloorData()

    # LIST
    entity.list(auth)

    # VIEW
    entity.view(2)

    # SEARCH
    entity.search('label=CrowdIndex&floorId=1', auth)

    # LATEST_BY_PROJECT
    entity.latest_by_project(projectId=1, auth=auth)

    # LATEST_BY_PROJECT_AND_LABEL
    label = "CrowdIndex"
    entity.latest_by_project_and_label(projectId=1, label=label, auth=auth)

    # CREATE
    data = {'floorId': '1', 'label': 'CrowdIndex', 'value': '60'}
    r = entity.create(data, auth)

    if r.status_code == 201:

        # UPDATE
        obj = r.json()
        obj['label'] = 'CrowdIndex'
        obj['value'] = '22'
        r2 = entity.update(obj, auth)

        # DELETE
        r3 = entity.delete(obj['id'], auth)

