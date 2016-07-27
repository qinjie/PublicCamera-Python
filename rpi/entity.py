from rpi_utils import get_logger

__author__ = 'zqi2'
import ConfigParser
import json

import requests
from requests.auth import HTTPBasicAuth


class Entity:
    # configuration file and section name
    _config_file = 'server.ini'
    # use <country> for testing purpose
    _config_section = 'default'

    # web service URLs
    _base_url = ''
    _urls = {}

    def __init__(self):
        self.logger = get_logger(name='entity', reset_handlers=True, log_file='batch.log', log_console=True)
        self.read_config()

    # Read settings from config file
    def read_config(self):
        # if '_urls' in globals():
        #     self._urls = globals()['_urls']
        #     return
        parser = ConfigParser.SafeConfigParser()
        parser.read(self._config_file)
        parser.defaults()
        self._base_url = parser.get('default', 'url_base')

        for key, val in parser.items(self._config_section):
            self._urls[key] = self._base_url + parser.get(self._config_section, key)

        self._base_url = parser.get('default', 'url_base')


    def list(self, auth=None):
        try:
            url = self._urls['list']
            headers = {'Accept': 'application/json'}
            r = requests.get(url, auth=auth, headers=headers)
            self.logger.info("LIST %s", url)
            self.logger.info("%s %s", r.status_code, r.headers['content-type'])
            self.logger.info(r.text)
            return r
        except requests.exceptions.RequestException as e:
            self.logger.error("Exception: " + str(e.message))
            return None

    def view(self, data_id, auth=None):
        try:
            url = self._urls['view'].replace("<id>", str(data_id))
            headers = {'Accept': 'application/json'}
            r = requests.get(url, auth=auth, headers=headers)
            self.logger.info("VIEW: %s", url)
            self.logger.info("%s %s", r.status_code, r.headers['content-type'])
            self.logger.info(r.text)
            return r
        except requests.exceptions.RequestException as e:
            self.logger.error("Exception: " + str(e.message))
            return None

    def search(self, query, auth=None):
        try:
            url = self._urls['search'].replace("<query>", str(query))
            headers = {'Accept': 'application/json'}
            r = requests.get(url, auth=auth, headers=headers)
            self.logger.info("SEARCH: %s", url)
            self.logger.info("%s %s", r.status_code, r.headers['content-type'])
            self.logger.info(r.text)
            return r
        except requests.exceptions.RequestException as e:
            self.logger.error("Exception: " + str(e.message))
            return None

    def create(self, payload, auth=None):
        try:
            url = self._urls['create']
            data = json.dumps(payload)
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
            r = requests.post(url, auth=auth, data=data, headers=headers)
            self.logger.info("CREATE: %s", url)
            self.logger.info("Payload = %s", data)
            self.logger.info("%s %s", r.status_code, r.headers['content-type'])
            self.logger.info(r.text)
            return r
        except requests.exceptions.RequestException as e:
            self.logger.error("Exception: " + str(e.message))
            return None

    def update(self, payload, auth=None):
        try:
            url = self._urls['update']
            id = payload['id']
            url = url.replace("<id>", str(id))
            # if 'id' in payload: del payload['id']

            data = json.dumps(payload)
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
            r = requests.put(url, data=data, auth=auth, headers=headers)
            self.logger.info("UPDATE: %s", url)
            self.logger.info("Payload = %s", data)
            self.logger.info("%s %s", r.status_code, r.headers['content-type'])
            self.logger.info(r.text)
            return r
        except requests.exceptions.RequestException as e:
            self.logger.error("Exception: " + str(e.message))
            return None

    def delete(self, data_id, auth=None):
        try:
            url = self._urls['delete']
            url = url.replace("<id>", str(data_id))
            r = requests.delete(url, auth=auth)
            self.logger.info("DELETE: %s", url)
            self.logger.info("%s %s", r.status_code, r.headers['content-type'])
            self.logger.info(r.text)
            return r
        except requests.exceptions.RequestException as e:
            self.logger.error("Exception: " + str(e.message))
            return None


if __name__ == '__main__':

    # # Read options from command line
    # argParser = argparse.ArgumentParser('API Entity')
    # argParser.add_argument('-c', '--configFile', help="Configuration file", required=False)
    # argParser.add_argument('-s', '--configSession', help="Configuration session", required=False)
    # argParser.add_argument('-u', '--username', help="Username", required=False)
    # argParser.add_argument('-p', '--password', help="Password", required=False)
    # args = argParser.parse_args()

    # Username and Password for Authentication
    username = 'user1'
    password = '123456'
    auth = HTTPBasicAuth(username, password)

    entity = Entity()

    # LIST
    entity.list(auth)

    # VIEW
    entity.view(4)

    # SEARCH
    entity.search('code=CN', auth)

    # CREATE
    data = {'code': 'CD', 'name': 'cdcdcd', 'population': '223344'}
    r = entity.create(data, auth)

    if r.status_code == 201:
        # UPDATE
        obj = r.json()
        obj['name'] = 'cd2cd2cd2'
        obj['population'] = '222333'
        r2 = entity.update(obj, auth)

        # DELETE
        r3 = entity.delete(obj['id'], auth)
