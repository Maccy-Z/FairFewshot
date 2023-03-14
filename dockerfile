FROM pytorch/pytorch:latest
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y python-pip
RUN pip install matplotlib scipy pandas
# RUN pip3 install reqs tensorboard black scipy
#RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv  \
#    torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
# RUN pip install pandas


