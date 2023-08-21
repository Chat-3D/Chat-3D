# ========================= data ==========================
anno_root = "anno"  # annotation dir
pc_encoder = "ulip2"  #
feat_root = f"/root/autodl-tmp/scene-LLaMA/datasets/referit3d/pcd_feats_{pc_encoder}"  # data dir for extracted features from pretrained 3d encoder
train_instance_file = f"{anno_root}/scannet_train_instances.json"
val_instance_file = f"{anno_root}/scannet_val_instances.json"
train_file_s1 = [
    [
        feat_root,
        train_instance_file,
        f"{anno_root}/scanrefer_train_captions_noun.json",
    ],
]
train_file_s2 = [
    [
        feat_root,
        train_instance_file,
        f"{anno_root}/scanrefer_train_stage2.json",
    ]
]
train_file_s3 = [
    [
        feat_root,
        train_instance_file,
        f"{anno_root}/scanrefer_train_conversation.json",
    ],
    [
        feat_root,
        train_instance_file,
        f"{anno_root}/scanrefer_train_describe.json"
    ]
]
val_file_s1 = [
    [
        feat_root,
        val_instance_file,
        f"{anno_root}/scanrefer_val_captions_noun.json",
    ]
]
val_file_s2 = [
    [
        feat_root,
        val_instance_file,
        f"{anno_root}/scanrefer_val_concise.json"
    ]
]
val_file_s3 = [
    [
        feat_root,
        val_instance_file,
        f"{anno_root}/scanrefer_val_describe100.json"
    ],
]


test_types = []
num_workers = 6

stop_key = None

# ========================= input ==========================
s1_batch_size = 12
s2_batch_size = 12
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
    system="###System: In this task, you have access to comprehensive information about objects within a 3D scene. Specifically, the focus is on the target object, which is identified between the tags \"<Target>\" and \"</Target>\". Additionally, the details of all the objects present in the scene can be found between the tags \"<Scene>\" and \"</Scene>\". <Target><TargetHere></Target> <Scene><SceneHere></Scene>",
    prompt_template="###Human: {} ###Assistant: ",
    max_txt_len=32,
    end_sym="###",
    stage=2
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
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
# eval_freq = 500
seed = 42

save_latest = False
auto_resume = True
pretrained_path = ""
