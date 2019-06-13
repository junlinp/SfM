FROM ubuntu:16.04
RUN apt-get update
RUN apt-get install -y cmake make git libgoogle-glog-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev
RUN git clone https://ceres-solver.googlesource.com/ceres-solver
WORKDIR /ceres-solver
RUN cmake . && make -j3 && make install

RUN git clone https://github.com/junlinp/SfM.git
RUN mkdir -p /SfM/cmake-build-debug
WORKDIR /Sfm/cmake-build-debug
RUN cmake ..
RUN make

CMD ["SfM"]