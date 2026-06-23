# Deploying an Open-Source VLM as a GPT Model Replacement

Here, we will explain how to deploy an open-source VLM on our own to replace the GPT family of models used by the agents in this benchmark. It is important to note that most open-source VLMs support only a single image input (e.g., `glm4v-9b-chat`). However, some agents require reasoning with multiple screenshots (e.g., `MobileAgentV2`), and the automatic evaluation of this benchmark also requires reasoning with multiple images. Therefore, it is recommended to use only the `exec` module of `AppAgent`.

## Setup

Run the setup script to initialize the environment and install necessary dependencies:

```bash
chmod +x ./setup.sh
./setup.sh
```

## Deploy

Select the model you want to use below and run the corresponding command. More models can be found on [this page](https://github.com/modelscope/swift/blob/main/docs/source_en/LLM/Supported-models-datasets.md#mllm).

```bash
# glm4v-9b-chat
conda activate swift
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type glm4v-9b-chat --host '0.0.0.0' --port 7001


# internlm-xcomposer2-7b-chat
conda activate swift
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type internlm-xcomposer2-7b-chat --host '0.0.0.0' --port 7001


# internlm-xcomposer2_5-7b-chat
conda activate swift
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type internlm-xcomposer2_5-7b-chat --host '0.0.0.0' --port 7001


# minicpm-v-v2_6-chat
conda activate swift
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type minicpm-v-v2_6-chat --host '0.0.0.0' --port 7001


# qwen-vl-chat
conda activate swift
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-vl-chat --host '0.0.0.0' --port 7001

```

#### Test

```bash
curl http://localhost:7001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm4v-9b-chat",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }
```

## Usage

1. Set `OPENAI_BASE_URL` to `"http://localhost:7001/v1"` in `Smartphone-Agent-Benchmark/.env`.
2. Set `OPENAI_API_MODEL` to `"glm4v-9b-chat"` in `Smartphone-Agent-Benchmark/config.yaml` (Note: you should change `"glm4v-9b-chat"` to the model you have actually deployed).
3. Run `python run.py --agent AppAgent --task_id cainiaoguoguo_0 --mode exec --session_id test` to test.
