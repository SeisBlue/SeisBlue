FROM tensorflow/tensorflow:2.0.0-gpu-py3-jupyter

RUN apt-get update && apt-get upgrade -y && apt-get install sudo openssh-server vim htop screen reptyr git -y
RUN echo '%staff   ALL=(ALL:ALL) ALL' >> /etc/sudoers
RUN pip install --upgrade pip

ENV TFX_DIR "/tfx"
ENV AIRFLOW_HOME "/airflow"
ENV SLUGIFY_USES_TEXT_UNIDECODE "yes"

RUN python -m pip --no-cache-dir install --upgrade tfx apache-airflow docker
RUN python -m pip --no-cache-dir install --upgrade obspy scikit-learn sklearn tqdm

RUN mkdir /var/run/sshd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# Delete Tensorflow login message for sftp login
RUN mv /etc/bash.bashrc /etc/bash.bashrc.old
RUN touch /etc/bash.bashrc
ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
