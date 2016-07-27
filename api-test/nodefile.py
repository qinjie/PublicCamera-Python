__author__ = 'zqi2'

import ConfigParser
import os

import requests
from requests.auth import HTTPBasicAuth

from entity import Entity


class NodeFile(Entity):
    # config section
    _config_section = 'node_file'
    _config_url_pre = 'url_'

    def __init__(self):
        Entity.__init__(self)

        parser = ConfigParser.SafeConfigParser()
        parser.read(self._config_file)
        for key, val in parser.items(self._config_section):
            temp = key.replace(self._config_url_pre, '')
            self._urls[temp] = self._base_url + parser.get(self._config_section, key)
            # self._urls['upload'] = self._base_url + parser.get(self._config_section, 'url_upload')
            # self._urls['latest_by_project'] = self._base_url + parser.get(self._config_section, 'url_latest_by_project')
            # self._urls['latest_by_project_and_label'] = self._base_url + parser.get(self._config_section,
            #                                                                         'url_latest_by_project_and_label')

    def upload(self, payload, auth=None):

        # -- Check if fileName is set
        if ('fileName' not in payload):
            self.log.error("fileName or fileType not set in payload.")
            return False
        # -- Check if file exists
        if not os.path.isfile(payload['fileName']):
            self.log.error("File not found: %s", payload['fileName'])
            return False
        # -- Set the 'file' form-data
        if 'fileType' in payload:
            files = {'file': (payload['fileName'], open(payload['fileName'], 'rb'), payload['fileType'])}
        else:
            files = {'file': (payload['fileName'], open(payload['fileName'], 'rb'))}

        url = self._urls['upload']
        headers = {'Accept': 'application/json'}
        r = requests.post(url, files=files, data=payload, auth=auth, headers=headers)
        self.log.info("UPLOAD: %s", url)
        self.log.info("%s %s", r.status_code, r.headers['content-type'])
        self.log.info(r.text)
        return r

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
        self.log.info("LATEST_BY_PROJECT_AND_LABEL: %s", url)
        self.log.info("%s %s", r.status_code, r.headers['content-type'])
        self.log.info(r.text)
        return r

    def delete_hours_older(self, hours, auth=None):

        url = self._urls['delete_hours_older'].replace("<hours>", str(hours))
        headers = {'Accept': 'application/json'}
        r = requests.delete(url, auth=auth, headers=headers)
        self.log.info("DELETE_HOURS_OLDER: %s", url)
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

    entity = NodeFile()

    # # LIST
    # entity.list(auth)
    #
    # # VIEW
    # entity.view(4)
    #
    # # SEARCH
    # entity.search('nodeId=3', auth)
    #
    # # LATEST_BY_PROJECT
    # entity.latest_by_project(projectId=1, auth=auth)
    #
    # # LATEST_BY_PROJECT_AND_LABEL
    # label = "This is your label"
    # entity.latest_by_project_and_label(projectId=1, label=label, auth=auth)

    # UPLOAD
    payload = {'nodeId': 4, 'fileName': 'testfiles/0002_20160119_000911_81238700.jpg', 'label': 'This is Test 4'}

    r = entity.upload(payload, auth)

    if r.status_code == 200:
        obj = r.json()

        # DELETE
        # r3 = entity.delete(obj['id'], auth)

        # DELETE_HOURS_OLDER

        # DELETE_HOURS_OLDER (Only manager or admin Allowed)

# username = 'manager1'
#    password = '123456'
#    manager = HTTPBasicAuth(username, password)
#    entity.delete_hours_older(hours=48, auth=manager)
