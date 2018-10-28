FROM tensorflow/tensorflow:1.9.0-gpu-py3

# Set the working directory to /workspace
WORKDIR /workspace

RUN apt-get update && apt-get install -y git build-essential swig pkg-config libopencv-dev libav-tools libjpeg-dev libpng-dev libtiff-dev libjasper-dev

RUN pip3 install --no-cache-dir Cython && pip install --upgrade pip

# Install the required libraries
COPY requirements.txt ./
RUN pip3 --no-cache-dir install -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run jupyter when container launches
CMD ["/bin/bash"]
