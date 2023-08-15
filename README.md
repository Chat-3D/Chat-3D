# Chat-3D 

This is a repo for paper "Chat-3D: Data-efficiently Tuning Large Language Model for Universal Dialogue of 3D Scenes". [ [paper](https://arxiv.org/abs/) / [project page](https://chat-3d.github.io/) ]

# Schedule

- [ ] Guide for data preparation and model training/inference
- [ ] High-quality object-centric instruction dataset 
- [ ] Online demo
- [ ] Try different settings of training data and model architecture
- [ ] ...

# Setup and Download


- Prepare the environment.
  ```shell
  pip install -r requirements.txt
  ```
  
- Download LLaMA model:
  - Currently, we choose 
Vicuna-7B as the LLM in our model, which is finetuned from LLaMA-7B.
  - Download LLaMA-7B from [hugging face](https://huggingface.co/decapoda-research/llama-7b-hf).
  - Download [vicuna-7b-delta-v0](https://huggingface.co/lmsys/vicuna-7b-delta-v0) and process it: (The `apply_delta.py` is from [huggingface](https://huggingface.co/CarperAI/stable-vicuna-13b-delta/raw/main/apply_delta.py))
  
  ```shell
  python3 model/apply_delta.py \
    --base /path/to/model_weights/llama-7b \
    --target vicuna-7b-v0 \
    --delta lmsys/vicuna-7b-delta-v0
  ```

  - Change the `llama_model_path` in [config.py](./scripts/config.py) to the location of `vicuna-7b-v0`.
  


# Three-stage Training

- 

# Inference

- 

# Citation

If you find this project useful in your research, please consider cite:
```BibTeX

```

# Acknowledgement

Thanks to the open source of the following projects:

[VideoChat](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat), [LLaMA](https://github.com/facebookresearch/llama), [ULIP](https://github.com/salesforce/ULIP), [ScanRefer](https://github.com/daveredrum/ScanRefer)