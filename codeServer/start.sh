#!/bin/bash

mkdir /root/jupyter_workspace
touch /root/jupyter_workspace/welcome.md
echo "hello" >> /root/jupyter_workspace/welcome.md

python edit_config.py
exec dumb-init /usr/bin/code-server /root/jupyter_workspace
