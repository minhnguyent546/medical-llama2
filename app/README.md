# Medical LLaMA 2 demo app

You can try out the demo in HuggingFace Spaces [here](https://huggingface.co/spaces/minhnguyent546/Med-Alpaca-2-7b-chat).

Or you can run it locally with Docker:
```bash
docker build \
    -t medical-llama2-app \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    . 

mkdir -p .cache/huggingface
docker run \
    -t \
    --rm \
    --publish 7860:7860 \
    -v ./.cache/huggingface:/home/user/app/.cache/huggingface \
    medical-llama2-app
```

Point your browser to http://localhost:7860 to access the app.
