FROM ubuntu:16.04
RUN apt-get update
RUN apt-get install -y cmake make git libgoogle-glog-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev
RUN git clone https://github.com/junlinp/SfM.git
RUN cd SfM
RUN mkdir -p cmake-build-debug
RUN cd cmake-build-debug
RUN git clone https://ceres-solver.googlesource.com/ceres-solver
RUN cd ceres-solver
RUN cmake ..
RUN make -j
RUN make test
RUN make install
RUN cd ..

RUN cmake ..
RUN make
CMD ["SfM"]