FROM pytorch/pytorch:latest

WORKDIR /app
COPY ResNet18.py /app/ResNet18.py

RUN pip install torchvision \
    && pip install matplotlib

ENTRYPOINT [ "python", "/app/ResNet18.py" ]
