
Setup Python env

```
sudo apt update
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y \
  python3.10 \
  python3.10-venv

python3.10 -m venv ~/.venv
source ~/.venv/bin/activate
```

Convert to Llama.cpp

```
pip install \
  torch \
  git+https://github.com/huggingface/transformers.git \
  git+https://github.com/huggingface/peft.git \
  sentencepiece

git clone https://github.com/ggerganov/llama.cpp
make

git clone https://huggingface.co/huggyllama/llama-30b huggyllama/llama-30b
find huggyllama/llama-30b/ | grep safetensors | xargs rm

git clone https://huggingface.co/IlyaGusev/saiga_30b_lora IlyaGusev/saiga_30b_lora

python ~/scripts/convert_pth.py \
  huggyllama/llama-30b \
  IlyaGusev/saiga_30b_lora \
  saiga_30b

python ~/src/llama.cpp/convert.py \
  --outtype f16 \
  --vocab-dir IlyaGusev/saiga_30b_lora \
  saiga_30b

rm saiga_30b/consolidated.00.pth

~/src/llama.cpp/quantize \
  saiga_30b/ggml-model-f16.bin \
  saiga_30b/ggml-model-q4_1.bin \
  3

```
