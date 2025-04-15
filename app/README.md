# Medical LLaMA 2 demo app

You can try out the demo in HuggingFace Spaces [here](https://huggingface.co/spaces/minhnguyent546/Med-Alpaca-2-7b-chat).

Or you can run it locally with Docker:
```bash
docker build -t medical-llama2-app . 
docker run --detach --rm --publish 7860:7860 medical-llama2-app
```

Point your browser to http://localhost:7860 to access the app.

**Note:** that the downloaded model will be cached in `./hf_cache` (in docker container), so make sure you have enough disk space.
