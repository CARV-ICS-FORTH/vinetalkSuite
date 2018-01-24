#!/bin/bash

#pkill -f "jobgen"
pkill -f "./build/vine_controller"
pkill -f "accelProcess"
pkill -f "monitoringThread"
pkill -f "jobgen"
rm -rf /dev/shm/

