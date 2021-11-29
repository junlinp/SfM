FROM ubuntu:18.04
# set up time zone
ENV TZ=US
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y libopencv-dev
RUN apt-get install -y cmake make git libgoogle-glog-dev  libeigen3-dev wget libgtest-dev

# link eigen
RUN ln -s /usr/include/eigen3/Eigen /usr/include/Eigen

# install ceres
RUN wget http://ceres-solver.org/ceres-solver-2.0.0.tar.gz && tar zxf ceres-solver-2.0.0.tar.gz
RUN apt-get install -y libatlas-base-dev
RUN apt-get install -y libsuitesparse-dev
WORKDIR /ceres-solver-2.0.0
RUN cmake . && make && make install


ADD ./src /SfM/src
ADD ./CMakeLists.txt /SfM/CMakeLists.txt
WORKDIR /SfM
RUN mkdir -p /SfM/cmake-build-release
WORKDIR /SfM/cmake-build-release
RUN cmake -DCMAKE_BUILD_TYPE=Release ..
RUN make
#CMD ["SfM"]