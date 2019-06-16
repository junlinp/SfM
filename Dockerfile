FROM dymat/opencv
RUN apt-get update
RUN apt-get install -y cmake make git libgoogle-glog-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev libsqlite3-dev wget
RUN git clone https://github.com/google/googletest.git
RUN wget http://ceres-solver.org/ceres-solver-1.14.0.tar.gz && tar zxf ceres-solver-1.14.0.tar.gz
RUN git clone https://github.com/junlinp/SfM.git

WORKDIR /googletest
RUN cmake . && make && make install

WORKDIR /ceres-solver-1.14.0
RUN cmake . && make && make install
RUN ln -s /usr/include/eigen3/Eigen /usr/include/Eigen

WORKDIR /SfM
RUN mkdir -p /SfM/cmake-build-debug
WORKDIR /SfM/cmake-build-debug
RUN cmake ..
RUN make

CMD ["SfM"]