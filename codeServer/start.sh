#!/bin/bash
cd ~
mkdir jupyter_workspace

# to do, wget the zip file and unzip the contents to jupyter workspace
# we use the following 2 rows as example instead
touch main.py
mv main.py jupyter_workspace/
mv /root/welcome.md jupyter_workspace/welcome.md

# apt-get update
# apt-get install expect -y
# apt-get install jq -y
jupyter lab --generate-config 

#使用expect处理交互式命令行程序
expect -c "
spawn jupyter lab password
expect \"Enter password:\"
send \"$USER_PASSWORD\r\"
expect \"Verify password:\"
send \"$USER_PASSWORD\r\"
expect eof
"
sleep 0.5
# Copy hashed password
hashed_password=$(jq -r '.IdentityProvider.hashed_password' ~/.jupyter/jupyter_server_config.json)
# echo $hashed_password

# Update Jupyter Lab config file with the hashed password
sed -i "s|^# c.ServerApp.password = .*|c.ServerApp.password = '$hashed_password'|" ~/.jupyter/jupyter_lab_config.py
cd jupyter_workspace
# Run Jupyter Lab，加上&使命令在后台执行而不阻塞当前脚本
jupyter lab --notebook-dir jupyter_workspace --ip 0.0.0.0 --port 8888 --allow-root --no-browser &

#启动code server
python edit_config.py
exec dumb-init /usr/bin/code-server /root/jupyter_workspace
