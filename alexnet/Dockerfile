FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
RUN python -m pip install --upgrade pip \
&& pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 


WORKDIR /app
COPY Alexnet.py /app/Alexnet.py

ENTRYPOINT [ "python", "-u","/app/Alexnet.py" ]
