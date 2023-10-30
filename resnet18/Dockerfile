FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY ResNet18.py /app/ResNet18.py

RUN pip install torchvision \
    && pip install matplotlib

ENTRYPOINT [ "python", "/app/ResNet18.py" ]
