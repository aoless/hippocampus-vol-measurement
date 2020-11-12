#!/bin/bash

# This script sends a study to the Orthanc server

# In your test data directory you will find three different studies - you may change the dir here
# to try all three out
storescu localhost 4242 -v -aec HIPPOAI +r +sd /data/TestVolumes/Study3
