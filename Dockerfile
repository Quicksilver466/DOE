FROM nvidia/cuda:12.5.1-base-ubuntu22.04
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN rm -f /etc/apt/sources.list.d/*.
RUN apt-get update && apt-get install && apt-get -y install curl && apt-get -y install git-all && apt-get -y install python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code
ENTRYPOINT ["/code/entrypoint.sh"]