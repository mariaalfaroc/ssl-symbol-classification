FROM pytorch/pytorch:latest

RUN apt update
RUN apt-get update -y
RUN apt install ffmpeg libsm6 -y
RUN apt install vim -y
RUN apt install nano

RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install sklearn
RUN pip install scikit-image
RUN pip install tqdm
RUN pip install torchinfo
RUN pip install pandas
