FROM python:3.8-bullseye

RUN apt-get -y update && \
	DEBIAN_FRONTEND=noninteractive \
	apt-get -y --no-install-recommends install build-essential git curl postgresql postgresql-client nano lsof npm zsh ffmpeg

RUN pip install --upgrade pip
# RUN locale-gen en_US.UTF-8
RUN mkdir commandhistory

RUN groupadd vscode
RUN useradd -g vscode vscode

# COPY args interpreted relative to where execute docker run, not where Dockerfile located.
ADD . /app

WORKDIR /app
RUN pwd && ls && pip install --no-cache-dir --user -r requirements.txt

CMD ["bash"]