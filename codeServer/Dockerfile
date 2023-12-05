FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
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
COPY welcome.md /root

COPY start.sh edit_config.py requirements.txt /root
COPY config.yaml /root/.config/code-server/
RUN pip install -r requirements.txt \
&& chmod 777 start.sh
CMD ["./start.sh"]
