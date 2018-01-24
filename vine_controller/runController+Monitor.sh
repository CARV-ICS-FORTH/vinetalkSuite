#!/bin/bash
./build/vine_controller example_config &>results/2.inputPowerTimers &
./monitoringTool/monitoringThread 60000 4 &> results/1.system_Info 
