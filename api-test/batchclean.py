__author__ = 'zqi2'

from requests.auth import HTTPBasicAuth

from nodefile import NodeFile

# Only manager or admin Allowed
username = 'manager1'
password = '123456'
manager = HTTPBasicAuth(username, password)

# Clean NodeFile
entity = NodeFile()
HOURS = 1200
entity.delete_hours_older(hours=HOURS, auth=manager)

