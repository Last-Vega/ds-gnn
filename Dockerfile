FROM nvcr.io/nvidia/pytorch:23.09-py3

WORKDIR /app
COPY . /app

# RUN conda install pytorch_geometric
# RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
# RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
# RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
# RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
# RUN pip install torch-geometric
RUN pip install torch_geometric
RUN pip install -r requirements.txt

