FROM ubuntu:18.04
# set up time zone
ENV TZ=US
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y libopencv-dev
RUN apt-get install -y cmake make git libgoogle-glog-dev  libeigen3-dev wget libgtest-dev

RUN wget http://ceres-solver.org/ceres-solver-2.0.0.tar.gz && tar zxf ceres-solver-2.0.0.tar.gz

#RUN git clone https://github.com/junlinp/SfM.git
COPY src /SFM/src
COPY CMakeLists.txt /SFM/CMakeLists.txt

RUN ln -s /usr/include/eigen3/Eigen /usr/include/Eigen

RUN apt-get install -y libatlas-base-dev
RUN apt-get install -y libsuitesparse-dev
WORKDIR /ceres-solver-2.0.0
RUN cmake . && make && make install


WORKDIR /SfM

RUN mkdir -p /SfM/cmake-build-debug

WORKDIR /SfM/cmake-build-debug
RUN cmake -DCMAKE_BUILD_TYPE=DEBUG ..
RUN make
#CMD ["SfM"]