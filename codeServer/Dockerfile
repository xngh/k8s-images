FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
RUN python -m pip install --upgrade pip \
&& pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
&& pip install jupyterlab \
&& apt-get update \
&& apt-get install expect -y \
&& apt-get install jq -y \
&& apt-get install -y \
	curl \
	dumb-init
RUN curl -fOL https://github.com/coder/code-server/releases/download/v4.19.0/code-server_4.19.0_amd64.deb \
  && dpkg -i code-server_4.19.0_amd64.deb
ENV PATH=/usr/local/bin/:$PATH \
    SHELL=/bin/bash
# COPY jupyter_lab_config.py /root/.jupyter/
COPY welcome.md edit_config.py /root/

COPY start.sh requirements.txt .
COPY config.yaml /root/.config/code-server/
RUN pip install -r requirements.txt 
RUN chmod 777 start.sh
CMD ["./start.sh"]
