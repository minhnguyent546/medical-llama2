FROM python:3.10-alpine

WORKDIR /app

RUN apk update && \
    apk add --no-cache \
    libgomp \
    libstdc++ \
    bash

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    grep -v llama-cpp-python requirements.txt | xargs pip install --no-cache-dir

# to avoid building from source, we install llama-cpp-python from a pre-built wheel
RUN pip install --no-cache-dir https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.2/llama_cpp_python-0.3.2-cp310-cp310-linux_x86_64.whl

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860
ENV no_proxy="localhost, 127.0.0.1, ::1"
EXPOSE 7860

CMD ["python", "app.py"]

COPY app.py app.py
