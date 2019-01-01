FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt update && apt install wget
RUN echo deb http://deb.obspy.org xenial main >> /etc/apt/sources.list
RUN wget --quiet -O - https://raw.github.com/obspy/obspy/master/misc/debian/public.key | apt-key add - 
RUN apt update

RUN apt install -y openssh-server python3-obspy
RUN mkdir /var/run/sshd

# Warning!!!! Please change the root password "screencast" below for security
RUN echo 'root:screencast' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]