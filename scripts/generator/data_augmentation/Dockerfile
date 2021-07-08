FROM python:3.6

RUN apt-get update --quiet
RUN apt-get install -y libblas-dev liblapack-dev libgeos-dev gfortran ffmpeg libsm6 libxext6

RUN pip install --upgrade pip
RUN pip install six numpy scipy
RUN pip install Pillow matplotlib scikit-image 
RUN pip install opencv-python
RUN pip install imageio Shapely
RUN pip install imgaug

RUN mkdir /imgaug
WORKDIR /imgaug

COPY augments.py /imgaug/augments.py

CMD python augments.py
