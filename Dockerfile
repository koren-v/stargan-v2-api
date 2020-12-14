FROM python:3.8-slim

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir /app
WORKDIR /app

RUN chmod +x ./download.sh

ADD requirements.txt /app
RUN pip3 install -r requirements.txt

ADD . /app

EXPOSE 5000
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["sh", "entrypoint.sh"]