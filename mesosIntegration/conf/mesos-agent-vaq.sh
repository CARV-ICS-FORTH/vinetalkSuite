#!/bin/bash
sudo rm -f $MESOS_HOME/meta/slaves/latest
ipaddr=127.0.0.1
"file=./resourcesWithVaq.txt"
if [ $# -eq 2  ]
  then
    ipaddr=$1
    file=$2
fi
if [ $# -eq 1 ]
  then
    ipaddr=$1
fi

sudo ../mesos/build/bin/mesos-agent.sh --master=$ipaddr:5050 --work_dir=$MESOS_HOME --resources=file://$file
