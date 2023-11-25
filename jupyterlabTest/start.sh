apt-get update
apt-get install expect
apt-get install jq
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

# Run Jupyter Lab
jupyter lab --notebook-dir jupyter_workspace --ip 0.0.0.0 --port 8888 --allow-root --no-browser 
