FROM tensorflow/tensorflow:2.2.0rc4-gpu-jupyter

RUN apt-mark hold libcudnn7

RUN apt-get update && apt-get upgrade -y
RUN apt-get install sudo openssh-server vim htop screen reptyr git ffmpeg libproj-dev proj-data proj-bin libgeos-dev -y

# Add all user into sudoers
RUN echo '%staff   ALL=(ALL:ALL) ALL' >> /etc/sudoers

RUN python -m pip install --upgrade pip
RUN python -m pip uninstall -y enum34; python -m pip --no-cache-dir install --upgrade jupyterlab pyyaml obspy scikit-learn sklearn tqdm Cartopy Pillow seaborn Sphinx sphinx-bootstrap-theme recommonmark

RUN mkdir /var/run/sshd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# Delete Tensorflow login message for sftp login
RUN mv /etc/bash.bashrc /etc/bash.bashrc.old
RUN touch /etc/bash.bashrc

ENV LD_LIBRARY_PATH /usr/local/cuda-10.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Add SeisNN into Python PATH
RUN echo "export PYTHONPATH=$PYTHONPATH:/SeisNN" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
