import os

def code_server_configuration():
  password = os.getenv("USER_PASSWORD")
  if not password:
      password = "123456"
  with open("/root/.config/code-server/config.yaml", "a") as f:
      f.write("\npassword: " + password)

code_server_configuration()
