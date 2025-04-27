FROM ubuntu:22.04 AS builder

RUN apt update && apt install -y --no-install-recommends \
    wget \
    ca-certificates

# install miniconda
RUN mkdir -p /opt/miniconda3 && \
    wget --no-verbose --show-progress \
    --progress=bar:force:noscroll \
    https://repo.anaconda.com/miniconda/Miniconda3-py39_25.1.1-2-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh && \
    bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 && \
    rm /opt/miniconda3/miniconda.sh

RUN /opt/miniconda3/bin/conda create -n medical-llama2 python=3.10 -y

# install requirements
WORKDIR /medical-llama2

COPY requirements.txt requirements.txt
ENV PATH="${PATH}:/opt/miniconda3/envs/medical-llama2/bin"
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------
FROM ubuntu:22.04

RUN apt update && apt install -y --no-install-recommends \
    ca-certificates

WORKDIR /medical-llama2

COPY --from=builder /opt/miniconda3/envs/medical-llama2/bin /opt/miniconda3/envs/medical-llama2/bin
COPY --from=builder /opt/miniconda3/envs/medical-llama2/lib /opt/miniconda3/envs/medical-llama2/lib

ENV PATH="${PATH}:/opt/miniconda3/envs/medical-llama2/bin"

ENTRYPOINT ["python"]

COPY . .
