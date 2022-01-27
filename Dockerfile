FROM armswdev/tensorflow-arm-neoverse:latest

RUN git clone https://github.com/swapnilvishwakarma/Person_Detection.git

WORKDIR /home/ubuntu/Person_Detection

EXPOSE 8080

CMD [ "python", "main.py", "-h" ]