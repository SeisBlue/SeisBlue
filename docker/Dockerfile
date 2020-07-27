FROM tensorflow/tensorflow:2.3.0rc2-gpu-jupyter

# nodejs source
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -

RUN apt-mark hold libcudnn7
RUN apt-get update && apt-get upgrade -y
RUN apt-get install sudo openssh-server vim htop screen reptyr git nodejs ffmpeg libproj-dev proj-data proj-bin libgeos-dev -y

# Add all user into sudoers
RUN echo '%staff   ALL=(ALL:ALL) ALL' >> /etc/sudoers

RUN python -m pip install --upgrade pip
RUN pip install shapely --no-binary shapely; python -m pip --no-cache-dir install --upgrade jupyterlab ipympl pyyaml obspy scikit-learn sklearn tqdm Cartopy Pillow seaborn pylint flake8 yapf Sphinx sphinx-bootstrap-theme recommonmark

# Jupyter Lab interactive plot
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build && jupyter labextension install jupyter-matplotlib --no-build
RUN jupyter lab build

# Change jupyter python kernel
RUN sed -i 's/\/usr\/bin\/python3/\/usr\/local\/bin\/python/g' /usr/local/share/jupyter/kernels/python3/kernel.json

RUN mkdir /var/run/sshd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# Delete Tensorflow login message for sftp login
RUN mv /etc/bash.bashrc /etc/bash.bashrc.old
RUN echo 'alias ls="ls --color"' > /etc/bash.bashrc

ENV LD_LIBRARY_PATH /usr/local/cuda-10.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Add SeisNN into Python PATH
RUN echo "export PYTHONPATH=$PYTHONPATH:/SeisNN" >> /etc/profile

# Change Jupyter interpreter
RUN sed -i 's/\/usr\/bin\/python3/\/usr\/local\/bin\/python/g' /usr/local/share/jupyter/kernels/python3/kernel.json

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
