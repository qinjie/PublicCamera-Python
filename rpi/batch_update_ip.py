import ConfigParser
import sys

from requests.auth import HTTPBasicAuth

from node import Node
from nodesetting import NodeSetting
from rpi_utils import get_ip_address, get_logger

__author__ = 'zqi2'

## ## Put scripts to /home/pi/python_batch folder. Edit the crontab to run the script at startup
## crontab -e
## ## Add following line to crontab so that it will run once after boot
## @reboot /usr/bin/python /home/pi/python_batch/batch_update_ip.py &

if __name__ == '__main__':

    # # Read options from command line
    # argParser = argparse.ArgumentParser('API Entity')
    # argParser.add_argument('-c', '--configFile', help="Configuration file", required=False)
    # argParser.add_argument('-s', '--configSession', help="Configuration session", required=False)
    # argParser.add_argument('-u', '--username', help="Username", required=False)
    # argParser.add_argument('-p', '--password', help="Password", required=False)
    # args = argParser.parse_args()

    # timestamp_file = 'last_job_timestamp.txt'
    # touch(timestamp_file)



    # configuration file and section name
    _config_file = 'batch.ini'
    # use <country> for testing purpose
    _config_section = 'default'

    parser = ConfigParser.SafeConfigParser()
    parser.read(_config_file)
    parser.defaults()

    FLOOR_ID = parser.get('default', 'floor_id')
    NODE_ID = parser.get('default', 'node_id')
    username = parser.get('default', 'username')
    password = parser.get('default', 'password')
    auth = HTTPBasicAuth(username, password)
    setting_interval = parser.get('default', 'interval_seconds')
    PHOTO_PATH = parser.get('default', 'photo_path')


    # Get the logger
    logger = get_logger(name='batch_update_ip', reset_handlers=True, log_file='log_update_ip.log', log_console=True)

    # Get node label
    node = Node()
    result = node.view(NODE_ID)
    if not result:
        logger.error("Invalid node ID {0}".format(NODE_ID))
        sys.exit()


    def update_ip():
        entity = NodeSetting()
        ip = get_ip_address('wlan0')
        data = {'nodeId': NODE_ID, 'label': 'wlan0_ip', 'value': ip}
        r = entity.create(data, auth)
        ip = get_ip_address('eth0')
        data = {'nodeId': NODE_ID, 'label': 'eth0_ip', 'value': ip}
        r = entity.create(data, auth)


    # update IP address of RPI for SSH
    update_ip()
