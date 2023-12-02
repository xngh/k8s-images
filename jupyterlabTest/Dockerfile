FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN python -m pip install --upgrade pip \
&& pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
&& pip install jupyterlab \
&& pip install torchvision \
&& apt-get update \
&& apt-get install expect -y \
&& apt-get install jq -y
ENV PATH=/usr/local/bin/:$PATH \
    SHELL=/bin/bash
# COPY jupyter_lab_config.py /root/.jupyter/
COPY start.sh .
COPY welcome.md /root
RUN chmod 777 start.sh
CMD ["./start.sh"]
