FROM continuumio/miniconda3

# install gcc and common build dependencies
RUN apt-get update \
 && apt-get install -y \
      build-essential \
      pylint

COPY environment.yml environment.yml

# install dependencies and fix tkinter error
# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
RUN conda env create --name eht-imaging --file environment.yml
RUN echo 'conda activate eht-imaging' >> ~/.bashrc
#RUN echo "backend: Agg" >> /opt/conda/lib/python3.9/site-packages/matplotlib/mpl-data/matplotlibrc

WORKDIR /eht-imaging
COPY . .