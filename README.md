# Chat-3D 

This is a repo for paper "Chat-3D: Data-efficiently Tuning Large Language Model for Universal Dialogue of 3D Scenes". 
[[paper](https://arxiv.org/abs/2308.08769)], [[project page](https://chat-3d.github.io/)]

### To Do List

**Done**

- [x] High-quality object-centric instruction dataset 
  
**Doing**

- [ ] Guide for data preparation and model training/inference locally
- [ ] Online demo
- [ ] Various settings of training data and model architecture
- [ ] Add a segmantation head to complete the pipeline
- [ ] ...

## 🔨 Preparation

- Prepare the environment:

  ```shell
  pip install -r requirements.txt
  ```
  
- Download LLaMA model:
  - Currently, we choose 
Vicuna-7B as the LLM in our model, which is finetuned from LLaMA-7B.
  - Download LLaMA-7B from [hugging face](https://huggingface.co/decapoda-research/llama-7b-hf).
  - Download [vicuna-7b-delta-v0](https://huggingface.co/lmsys/vicuna-7b-delta-v0) and process it: (`apply_delta.py` is from [huggingface](https://huggingface.co/CarperAI/stable-vicuna-13b-delta/raw/main/apply_delta.py))
  
  ```shell
  python3 model/apply_delta.py \
          --base /path/to/model_weights/llama-7b \
          --target vicuna-7b-v0 \
          --delta lmsys/vicuna-7b-delta-v0
  ```

  - Change the `llama_model_path` in [config.py](./scripts/config.py) to the location of `vicuna-7b-v0`.
  
- 3D point cloud and extracted features:

  - For training and inference on ScanRefer
  
  
- Object-centric dataset

  - We release the object-centric dataset in [annotations](./annotations) dir, including train/val sets for conversation/detail instructions.

## 🤖 Training and Inference

- Training (Instruction Tuning)
  
  Simply run the following scripts to sequancially tune from Stage 1 to Stage 3. 
  
  ```shell
  # Stage 1
  ./scripts/run.sh --stage 1 \
                   --lr 5e-3
  
  # Stage 2
  ./scripts/run.sh --stage 2 \
                   --pretrained_path /path/to/pretrained_stage1.pth \
                   --lr 5e-3
  
  # Stage 3
  ./scripts/run.sh --stage 3 \
                   --pretrained_path /path/to/pretrained_stage2.pth \
                   --lr 5e-5
  ```

- Inference
  
  - Use one GPU for inference (set `NUM_GPUS=1` in [run.sh](./scripts/run.sh)).
  ```shell
  ./scripts/run.sh --stage 3 \
                   --pretrained_path /path/to/pretrained_stage3.pth \
                   --evaluate
  ```

## 📄 Citation

If you find this project useful in your research, please consider cite:
```BibTeX
@misc{wang2023chat3d,
      title={Chat-3D: Data-efficiently Tuning Large Language Model for Universal Dialogue of 3D Scenes}, 
      author={Zehan Wang and Haifeng Huang and Yang Zhao and Ziang Zhang and Zhou Zhao},
      year={2023},
      eprint={2308.08769},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Stay tuned for our project. 🔥

If you have any questions or suggestions, feel free to drop us an email `wangzehan01@zju.edu.cn`, `huanghaifeng@zju.edu.cn` or open an issue.

## 😊 Acknowledgement

Thanks to the open source of the following projects:

[VideoChat](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat), [LLaMA](https://github.com/facebookresearch/llama), [ULIP](https://github.com/salesforce/ULIP), [ScanRefer](https://github.com/daveredrum/ScanRefer), [ReferIt3D](https://github.com/referit3d/referit3d), [vil3dref](https://github.com/cshizhe/vil3dref), [ScanNet](https://github.com/ScanNet/ScanNet) 