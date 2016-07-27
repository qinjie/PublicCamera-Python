__author__ = 'zqi2'
import ConfigParser
import json
import logging
import os
from urlparse import urlparse

import requests
from requests.auth import HTTPBasicAuth


class Entity:
    # logger
    log = None

    # configuration file and section name
    _config_file = 'server.ini'
    # use <country> for testing purpose
    _config_section = 'default'

    # web service URLs
    _base_url = ''
    _urls = {}

    def __init__(self):
        if 'log' not in globals():
            self.init_logger()
            self.log = logging.getLogger('main')
            globals()['log'] = self.log
        else:
            self.log = globals()['log']

        self.read_config()

    def init_logger(self):
        LOG_FILENAME = 'log'
        self.log = logging.getLogger('main')
        self.log.setLevel(logging.INFO)
        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

        fileHandler = logging.FileHandler("{0}/{1}.txt".format(os.getcwd(), LOG_FILENAME))
        fileHandler.setFormatter(log_formatter)
        self.log.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_formatter)
        self.log.addHandler(consoleHandler)

    # Read settings from config file
    def read_config(self):
        parser = ConfigParser.SafeConfigParser()
        parser.read(self._config_file)
        parser.defaults()
        self._base_url = parser.get('default', 'url_base')
        self._urls['view'] = self._base_url + parser.get(self._config_section, 'view')
        self._urls['list'] = self._base_url + parser.get(self._config_section, 'list')
        self._urls['create'] = self._base_url + parser.get(self._config_section, 'create')
        self._urls['update'] = self._base_url + parser.get(self._config_section, 'update')
        self._urls['delete'] = self._base_url + parser.get(self._config_section, 'delete')
        self._urls['search'] = self._base_url + parser.get(self._config_section, 'search')

        # validate URLs
        self.log.info('Reading urls from setting file')
        for _, url in self._urls.iteritems():
            parsed_url = urlparse(url)
            if not bool(parsed_url.scheme):
                self.log.error('Invalid URL: ' + url)
            else:
                # self.log.info('URL OK: ' + url)
                pass

    def list(self, auth=None):
        url = self._urls['list']
        headers = {'Accept': 'application/json'}
        r = requests.get(url, auth=auth, headers=headers)
        self.log.info("LIST %s", url)
        self.log.info("%s %s", r.status_code, r.headers['content-type'])
        self.log.info(r.text)
        return r

    def view(self, data_id, auth=None):
        url = self._urls['view'].replace("<id>", str(data_id))
        headers = {'Accept': 'application/json'}
        r = requests.get(url, auth=auth, headers=headers)
        self.log.info("VIEW: %s", url)
        self.log.info("%s %s", r.status_code, r.headers['content-type'])
        self.log.info(r.text)
        return r

    def search(self, query, auth=None):
        url = self._urls['search'].replace("<query>", str(query))
        headers = {'Accept': 'application/json'}
        r = requests.get(url, auth=auth, headers=headers)
        self.log.info("SEARCH: %s", url)
        self.log.info("%s %s", r.status_code, r.headers['content-type'])
        self.log.info(r.text)
        return r

    def create(self, payload, auth=None):
        url = self._urls['create']
        data = json.dumps(payload)
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        r = requests.post(url, auth=auth, data=data, headers=headers)
        self.log.info("CREATE: %s", url)
        self.log.info("Payload = %s", data)
        self.log.info("%s %s", r.status_code, r.headers['content-type'])
        self.log.info(r.text)
        return r

    def update(self, payload, auth=None):
        url = self._urls['update']
        id = payload['id']
        url = url.replace("<id>", str(id))
        # if 'id' in payload: del payload['id']

        data = json.dumps(payload)
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        r = requests.put(url, data=data, auth=auth, headers=headers)
        self.log.info("UPDATE: %s", url)
        self.log.info("Payload = %s", data)
        self.log.info("%s %s", r.status_code, r.headers['content-type'])
        self.log.info(r.text)
        return r

    def delete(self, data_id, auth=None):
        url = self._urls['delete']
        url = url.replace("<id>", str(data_id))
        r = requests.delete(url, auth=auth)
        self.log.info("DELETE: %s", url)
        self.log.info("%s %s", r.status_code, r.headers['content-type'])
        self.log.info(r.text)
        return r


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
