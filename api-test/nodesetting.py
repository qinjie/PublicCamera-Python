__author__ = 'zqi2'

import ConfigParser
import argparse
import requests
from requests.auth import HTTPBasicAuth
from entity import Entity

class NodeSetting (Entity):
    # config section
    _config_section = 'node_setting'

    def __init__(self):
        Entity.__init__(self)

        parser = ConfigParser.SafeConfigParser()
        parser.read(self._config_file)
        self._urls['update_ip'] = self._base_url + parser.get(self._config_section, 'url_update_ip')

    def update_ip(self, node_id, auth=None):
        url = self._urls['update_ip']
        url = url.replace("<node-id>", str(node_id))
        # if 'id' in payload: del payload['id']

        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        r = requests.put(url, auth=auth, headers=headers)
        self.log.info("UPDATE: %s", url)
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

    entity = NodeSetting()

    # UPDATE IP
    entity.update_ip(1, auth)

    # LIST
    entity.list(auth)

    # VIEW
    entity.view(1)

    # SEARCH
    entity.search('label=interval', auth)

    # CREATE
    data = {'nodeId': '1', 'label': 'sleep', 'value': '1'}
    r = entity.create(data, auth)

    if r.status_code == 201:

        # UPDATE
        obj = r.json()
        obj['label'] = 'sleep1'
        obj['value'] = '10'
        r2 = entity.update(obj, auth)

        # DELETE
        r3 = entity.delete(obj['id'], auth)

