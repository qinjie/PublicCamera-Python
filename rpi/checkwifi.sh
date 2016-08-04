sudo apt-get clean
ping -c4 153.20.9.97 > /dev/null

if [ $? != 0 ]
then
  echo "No network connection, restarting wlan0"
  /sbin/ifdown 'wlan0'
  sleep 5
  /sbin/ifup --force 'wlan0'
else
  echo "Network connection OK"
fi

