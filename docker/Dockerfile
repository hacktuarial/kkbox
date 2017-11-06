# Ubuntu image with python3, xgboost and diamond installed
FROM ubuntu

RUN apt-get update && \
apt-get install -y python3 gcc-4.9 tree python3-pip p7zip-full git

# set up python virtual environment
RUN pip3 install cython numpy pandas scipy click joblib sklearn dill tqdm

# install diamond
RUN git clone http://github.com/stitchfix/diamond.git && pip3 install -e diamond

# install xgboost per https://xgboost.readthedocs.io/en/latest/build.html
RUN git clone --recursive https://github.com/dmlc/xgboost && \
cd xgboost && \
make -j4 && \
pip3 install -e python-package && \
cd ..
