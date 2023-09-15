# ========================= data ==========================
anno_root = "annotations"  # annotation dir
pc_encoder = "ulip2"
feat_file = f"{anno_root}/scannet_{pc_encoder}_feats.pt"
attribute_file = f"{anno_root}/scannet_attributes_old.json"
train_file_s1 = [
    [
        feat_file,
        attribute_file,
        f"{anno_root}/scanrefer_train_stage1.json",
    ],
    # [
    #     feat_file,
    #     attribute_file,
    #     f"{anno_root}/nr3d_train_stage1.json",
    # ],
    # [
    #     feat_file,
    #     attribute_file,
    #     f"{anno_root}/scannet_train_stage1.json",
    # ],
    # [
    #     f"{anno_root}/objaverse_{pc_encoder}_feats.pt",
    #     f"{anno_root}/objaverse_attributes.json",
    #     f"{anno_root}/objaverse_stage1.json"
    # ]
]
train_file_s2 = [
    [
        feat_file,
        attribute_file,
        f"{anno_root}/scanrefer_train_stage2.json",
    ],
    # [
    #     feat_file,
    #     attribute_file,
    #     f"{anno_root}/nr3d_train_stage2.json"
    # ]
]
train_file_s3 = [
    [
        feat_file,
        attribute_file,
        f"{anno_root}/scanqa_train_stage3.json",
        1
    ],
    # [
    #     feat_file,
    #     attribute_file,
    #     f"{anno_root}/scanrefer_train_conversation.json",
    #     3
    # ],
    # [
    #     feat_file,
    #     attribute_file,
    #     f"{anno_root}/scanrefer_train_detail.json",
    #     1
    # ],
    # [
    #     feat_file,
    #     attribute_file,
    #     f"{anno_root}/nr3d_train_tf.json",
    #     1
    # ]
]
val_file_s1 = [
    [
        feat_file,
        attribute_file,
        f"{anno_root}/scanrefer_val_stage1.json",
    ]
]
val_file_s2 = [
    [
        feat_file,
        attribute_file,
        f"{anno_root}/scanrefer_val_stage2.json"
    ]
]
val_file_s3 = [
    [
        feat_file,
        attribute_file,
        f"{anno_root}/scanqa_val_gtobj.json"
    ],
]


test_types = []
num_workers = 6

stop_key = None

# ========================= input ==========================
s1_batch_size = 12
s2_batch_size = 8
s3_batch_size = 1
# max_txt_l = 32

pre_text = False


# ========================= model ==========================
model = dict(
    llama_model_path="model/vicuna-7b-v0",
    # gpt_model_path="",
    input_dim=512 if pc_encoder == "ulip2" else 4096,
    input_attr_dim=9,
    encoder_num_layers=1,
    mlp_dropout=0.1,
    low_resource=False,
    prompt_path="prompts/concise_description.txt",
    system_path="prompts/system.txt",
    prompt_template="###Human: {} ###Assistant: ",
    max_txt_len=512,
    end_sym="###",
    stage=3
)

optimizer = dict(
    opt="adamW",
    lr=5e-3,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=["module.color_size_proj", "module.scene_proj", "param module.input_norm"], lr=1e-5),
)

scheduler = dict(sched="cosine", epochs=6, min_lr_multi=0.01, warmup_epochs=0.2)

evaluate = False
deep_fusion = False

fp16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="noname_0",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="3DChat",  # setup in your command line
)
dist_url = "env://"
device = "cuda"

# ========================= others ==========================
output_dir = "outputs/tmp"  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
# eval_freq = 500
seed = 42

save_latest = False
auto_resume = True
pretrained_path = ""
