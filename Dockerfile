FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN rm -f /etc/apt/sources.list.d/*.
RUN apt-get update && apt-get install && apt-get -y install curl && apt-get -y install jq && apt-get -y install git-all && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code
ENTRYPOINT ["/code/entrypoint.sh"]