#pull official base image
FROM python:3.9.6

#set update
RUN apt-get update -y

# install dependencies
COPY ./requirements.txt /usr/requirements/
RUN pip install --upgrade pip && pip install -r /usr/requirements/requirements.txt
