# Reference  http://raspberrypi.stackexchange.com/questions/6714/how-to-get-the-raspberry-pis-ip-address-for-ssh

import fcntl
import logging
import os
import socket
import struct


def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        return socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', ifname[:15])
        )[20:24])
    except Exception:
        return None

def get_logger(name='main', reset_handlers=True, log_file=None, log_console=True, log_level=logging.INFO):
    logger = logging.getLogger(name)

    logger.setLevel(log_level)
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    # Clear handler if required
    if logger.handlers and reset_handlers:
        logger.handlers = []

    if log_file:
        fileHandler = logging.FileHandler("{0}/{1}.txt".format(os.getcwd(), log_file))
        fileHandler.setFormatter(log_formatter)
        logger.addHandler(fileHandler)

    if log_console:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_formatter)
        logger.addHandler(consoleHandler)

    return logger

if __name__ == '__main__':
    print get_ip_address('wlan0')
    print get_ip_address('eth0')
