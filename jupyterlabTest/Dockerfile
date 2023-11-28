FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN pip install jupyterlab \
&& pip install torchvision
ENV PATH=/usr/local/bin/:$PATH \
    SHELL=/bin/bash
# COPY jupyter_lab_config.py /root/.jupyter/
COPY start.sh .
COPY welcome.md /root
RUN chmod 777 start.sh
CMD ["./start.sh"]
