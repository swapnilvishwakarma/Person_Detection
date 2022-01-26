FROM armswdev/tensorflow-arm-neoverse:latest

WORKDIR /home/ubuntu/Person_Detection

RUN git clone https://github.com/swapnilvishwakarma/Person_Detection.git

CMD [ "python", "main.py", "-h" ]