#pull official base image
FROM python:3.9.6

#set update
# install dependencies
RUN apt-get update && apt-get install -y \
    tzdata \
    libgconf-2-4

#install chrome driver
RUN wget -P /tmp https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
&& apt-get update \
&& apt-get install -y --fix-broken /tmp/google-chrome-stable_current_amd64.deb

COPY ./requirements.txt /usr/requirements/

RUN pip install --upgrade pip && pip install -r /usr/requirements/requirements.txt

