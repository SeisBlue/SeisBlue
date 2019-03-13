# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ARG APT_INSTALL="apt-get install -y --no-install-recommends" 
ARG PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" 

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list 

RUN apt-get update 

# ==================================================================
# tools
# ------------------------------------------------------------------

RUN ${APT_INSTALL} \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim 

# ==================================================================
# python
# ------------------------------------------------------------------

RUN ${APT_INSTALL} \
        software-properties-common 
        
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

RUN ${APT_INSTALL} \
        python3.6 \
        python3.6-dev  
        
RUN wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py 

RUN python3.6 ~/get-pip.py 
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3 
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python 

RUN ${PIP_INSTALL} \
        setuptools
        
RUN ${PIP_INSTALL} \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-learn==0.20.0 \
        matplotlib \
        Cython 
        
# ==================================================================
# pytorch
# ------------------------------------------------------------------

RUN ${PIP_INSTALL} \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
    	torchvision_nightly \
        torchsummaryx 
        
RUN ${PIP_INSTALL} \
        torch_nightly -f \
        https://download.pytorch.org/whl/nightly/cu100/torch_nightly.html 
        
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig 
RUN apt-get clean 
RUN apt-get autoremove 
RUN rm -rf /var/lib/apt/lists/* /tmp/* ~/*


# ==================================================================
# ssh
# ------------------------------------------------------------------

RUN ${PIP_INSTALL} obspy
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:CHANGEPASSWORD' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
