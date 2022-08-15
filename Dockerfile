FROM intel/intel-optimized-pytorch:1.11.0-pip

ENV DEBIAN_FRONTEND="noninteractive"

WORKDIR /tmp

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY pizza /tmp/pizza
COPY setup.py /tmp/setup.py
RUN pip3 install .

#ENV PATH /home/root/.local/bin:${PATH}
WORKDIR /app
COPY main.py .

ENV PIZZA_MODEL_PATH /tmp/pizza/resnet18_pizza_cls_model.pth

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]
