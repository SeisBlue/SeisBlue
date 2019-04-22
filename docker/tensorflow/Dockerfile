FROM tensorflow/tensorflow:2.0.0a0-gpu-py3-jupyter

RUN apt-get update && apt-get upgrade -y && apt-get install openssh-server -y
RUN python -m pip --no-cache-dir install --upgrade obspy scikit-learn

RUN mkdir /var/run/sshd
RUN echo 'root:CHANGEPASSWORD' | chpasswd

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
