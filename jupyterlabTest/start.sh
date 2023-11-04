#!/bin/bash
cd ~
mkdir jupyter_workspace

# to do, wget the zip file and unzip the contents to jupyter workspace
# we use the following 2 rows as example instead
touch main.py
mv main.py jupyter_workspace/

jupyter lab --notebook-dir jupyter_workspace --ip 0.0.0.0 --port 8888 --allow-root --no-browser --ServerApp.password="" --ServerApp.token=""

