FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN useradd -l -m -s /bin/bash -N -u 1000 code

ENV HOME /home/code

COPY requirements.txt /opt/CodeGeeX2/requirements.txt

WORKDIR /opt/CodeGeeX2

RUN pip install -r requirements.txt 

RUN CMAKE_ARGS="-DGGML_CUBLAS=ON" pip install --use-deprecated=legacy-resolver -U chatglm-cpp

RUN pip uninstall transformer-engine -y

COPY --chown=1000:1000 . /opt/CodeGeeX2

RUN rm -rf ~/.cache