FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive


RUN apt update
RUN apt install -y python-pip


RUN pip install matplotlib scipy pandas scikit-learn
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
RUN pip install torch_geometric==2.3
RUN pip install openml
# RUN pip3 install reqs tensorboard black



