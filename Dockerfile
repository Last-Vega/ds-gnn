# FROM nvcr.io/nvidia/pytorch:23.01-py3
# FROM nvcr.io/nvidia/pytorch:22.08-py3
FROM nvcr.io/nvidia/pytorch:22.12-py3

# FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

WORKDIR /app
COPY . /app

# RUN conda install pytorch_geometric
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install torch-geometric
RUN pip install -r requirements.txt

