# Chat-3D 

This is a repo for paper "Chat-3D: Data-efficiently Tuning Large Language Model for Universal Dialogue of 3D Scenes". 
[[paper](https://arxiv.org/abs/2308.08769)], [[project page](https://chat-3d.github.io/)]

### News

[2023.12.15] 🔥 We release a new version: Chat-3D v2 [[code](https://github.com/Chat-3D/Chat-3D-v2), [paper](https://arxiv.org/abs/2312.08168)], achieving strong performance on various 3D scene-language tasks.

## 🔨 Preparation

- Prepare the environment:

  ```shell
  conda create -n chat-3d python=3.9.17
  conda activate chat-3d
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
  pip install -r requirements.txt
  ```
  
- Download LLaMA model:
  - Currently, we choose 
Vicuna-7B as the LLM in our model, which is finetuned from LLaMA-7B.
  - Download LLaMA-7B from [hugging face](https://huggingface.co/docs/transformers/main/model_doc/llama).
  - Download [vicuna-7b-delta-v0](https://huggingface.co/lmsys/vicuna-7b-delta-v0) and process it: (`apply_delta.py` is from [huggingface](https://huggingface.co/CarperAI/stable-vicuna-13b-delta/raw/main/apply_delta.py))
  
  ```shell
  python3 model/apply_delta.py \
          --base /path/to/model_weights/llama-7b \
          --target vicuna-7b-v0 \
          --delta lmsys/vicuna-7b-delta-v0
  ```

  - Change the `llama_model_path` in [config.py](./scripts/config.py) to the location of `vicuna-7b-v0`.
  

- Annotations and extracted features:

  **For simplicity, we have made all the annotations available in [annotations](./annotations) dir and extracted features available on [Google Drive](https://drive.google.com/drive/folders/1jQQFHeazZQpxKXFZXonrTu2HYSJRlA6T?usp=sharing).** Here are some brief explanations of the preparation:

  - Based on the annotations from [ScanNet](https://github.com/ScanNet/ScanNet) , we extract attributes (location, size, color) of objects from different scenes. 
  
  - We use [ULIP-2](https://github.com/salesforce/ULIP) to extract features of 3D objects. 
  
  - The captions utilized in stage 1 and stage 2 are obtained from the annotations of [ScanRefer](https://github.com/daveredrum/ScanRefer).
  
  
- Object-centric dataset

  - We release the object-centric dataset in [annotations](./annotations) dir, including train/val sets for conversation/detail instructions.

## 🤖 Training and Inference

- Training (Instruction Tuning)
  
  Simply run the following scripts to sequentially tune from Stage 1 to Stage 3. 
  
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
  
  We train the model on 4 `A40` GPUs with 48GB VRAM. Here are some information about GPU usage and training time. (Note that we only use [ScanRefer](https://github.com/daveredrum/ScanRefer) data for training currently, it would cost more training time if we add more training data in the future.)
  
  | Stage | Batch Size | GPU Num | VRAM Usage per GPU | Training Time |
  | --- | --- | --- | --- | --- |
  | 1 | 12 | 4 | ~ 25 GB | ~ 5 min |
  | 2 | 12 | 4 | ~ 45 GB | ~ 1 hour |
  | 3 | 1 | 4 | ~ 25 GB | ~ 1.5 hour |

- Inference
  
  Use one GPU for inference (set `NUM_GPUS=1` in [run.sh](./scripts/run.sh)).
  ```shell
  ./scripts/run.sh --stage 3 \
                   --pretrained_path /path/to/pretrained_stage3.pth \
                   --evaluate
  ```

## 📄 Citation

If you find this project useful in your research, please consider cite:
```BibTeX
@article{wang2023chat,
  title={Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes},
  author={Wang, Zehan and Huang, Haifeng and Zhao, Yang and Zhang, Ziang and Zhao, Zhou},
  journal={arXiv preprint arXiv:2308.08769},
  year={2023}
}
@article{huang2023chat,
  title={Chat-3D v2: Bridging 3D Scene and Large Language Models with Object Identifiers},
  author={Huang, Haifeng and Wang, Zehan and Huang, Rongjie and Liu, Luping and Cheng, Xize and Zhao, Yang and Jin, Tao and Zhao, Zhou},
  journal={arXiv preprint arXiv:2312.08168},
  year={2023}
}
```

Stay tuned for our project. 🔥

If you have any questions or suggestions, feel free to drop us an email (`wangzehan01@zju.edu.cn`, `huanghaifeng@zju.edu.cn`) or open an issue.

## 😊 Acknowledgement

Thanks to the open source of the following projects:

[VideoChat](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat), [LLaMA](https://github.com/facebookresearch/llama), [ULIP](https://github.com/salesforce/ULIP), [ScanRefer](https://github.com/daveredrum/ScanRefer), [ReferIt3D](https://github.com/referit3d/referit3d), [vil3dref](https://github.com/cshizhe/vil3dref), [ScanNet](https://github.com/ScanNet/ScanNet) 