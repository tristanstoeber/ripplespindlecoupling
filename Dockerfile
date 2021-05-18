FROM jupyter/datascience-notebook

USER root
RUN apt-get update && apt-get install -yq dvipng

USER jovyan

RUN pip install tqdm
RUN pip install quantities
RUN pip install ripple_detection
RUN pip install git+https://github.com/CINPLA/pyxona.git
RUN pip install odfpy








