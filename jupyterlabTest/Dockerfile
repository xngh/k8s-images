FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
#安装必要的库和工具
RUN apt-get update \
&& python -m pip install --upgrade pip \
&& pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
&& pip install jupyterlab \
&& pip install torchvision \
&& apt-get install expect -y \
&& apt-get install jq -y \
&& apt-get install -y openssh-server \
&& apt-get install -y net-tools \
&& apt-get install -y vim 

#配置ssh
RUN mkdir -p /var/run/sshd \
&& mkdir -p mkdir/root/.ssh/ \
&& echo root:123456 | chpasswd
ENV PATH=/usr/local/bin/:$PATH \
    SHELL=/bin/bash
# COPY jupyter_lab_config.py /root/.jupyter/
COPY start.sh .
COPY welcome.md /root
RUN chmod 777 start.sh
CMD ["./start.sh"]
