FROM ubuntu:16.04
RUN apt-get update
RUN apt-get install -y cmake make git
RUN git clone https://github.com/junlinp/SfM.git
RUN cd SfM
RUN mkdir -p cmake-build-debug
RUN cd cmake-build-debug
RUN cmake ..
RUN make
CMD ["SfM"]