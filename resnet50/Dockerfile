FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
RUN python -m pip install --upgrade pip \
&& pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
&& apt-get update 

WORKDIR /app
COPY Resnet50.py /app/Resnet50.py

ENTRYPOINT [ "python", "-u","/app/Resnet50.py" ]