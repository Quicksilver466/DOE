FROM python:3.11.6-slim
ENV PYTHONUNBUFFERED 1
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends netcat && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . /code
ENTRYPOINT [ "entrypoint.sh" ]