import random
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
from models.transformer_vanilla import TransformerEncoder
from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine

from transformers import StoppingCriteria, StoppingCriteriaList

import contextlib

logger = logging.getLogger(__name__)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


class Chat3D(nn.Module):
    """
    VideoChat model.
    """
    def __init__(self, config):
        super().__init__()
        llama_model_path = config.get("llama_model_path")
        low_resource = config.get("low_resource", False) # use 8 bit and put vit in cpu
        # prompt
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 32)
        end_sym = config.get("end_sym", '\n')
        self.system = config.get("system", "")
        self.begin_signal = "###"
        self.role = ("Human", "Assistant")
        self.pc_start_token, self.pc_end_token = "<Target>", "</Target>"
        self.scene_start_token, self.scene_end_token = "<Scene>", "</Scene>"

        mlp_dropout = config.get("mlp_dropout", 0.5)
        self.stage = config.get("stage", 1)

        self.low_resource = low_resource

        self.input_dim = config.get("input_dim", 4096)
        self.input_attr_dim = config.get("input_attr_dim", 9)

        logger.info('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
            )
        logger.info("freeze LLAMA")
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logger.info('Loading LLAMA Done')

        self.scene_proj = GenericMLP(
            input_dim=self.input_dim,
            hidden_dims=[self.llama_model.config.hidden_size],
            output_dim=self.llama_model.config.hidden_size,
            norm_fn_name="ln",
            output_use_activation=False,
            output_use_norm=True,
            output_use_bias=False,
            dropout=mlp_dropout
        )
        self.color_size_proj = nn.Linear(
            6, self.input_dim
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=self.llama_model.config.hidden_size, pos_type="fourier"
        )
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.llama_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.encoder_num_layers = config.get("encoder_num_layers", 1)
        self.relation_module = TransformerEncoder(dim=self.llama_model.config.hidden_size, num_layers=self.encoder_num_layers)

        if self.stage == 1:
            for p in self.relation_module.parameters():
                p.requires_grad = False

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            self.prompt_list = self.process_prompt(prompt_path, prompt_template)
        else:
            self.prompt_list = []

    def process_prompt(self, prompt_path, prompt_template):
        with open(prompt_path, 'r') as f:
            raw_prompts = f.read().splitlines()
        filted_prompts = [raw_prompt for raw_prompt in raw_prompts]
        prompt_list = [self.system + " " + prompt_template.format(p) for p in filted_prompts]
        logger.info(f'Load {len(prompt_list)} training prompts')
        logger.info(f'Prompt: {prompt_list}')
        return prompt_list

    def prompt_wrap(self, pc_embed, scene_embed, scene_attn, prompt, is_eval=False):
        batch_size = pc_embed.shape[0]
        p_0, p_ = prompt.split('<TargetHere>')
        p_1, p_2 = p_.split('<SceneHere>')
        p_0_tokens = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=is_eval).to(pc_embed.device)
        p_1_tokens = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False).to(pc_embed.device)
        p_2_tokens = self.llama_tokenizer(p_2, return_tensors="pt", add_special_tokens=False).to(pc_embed.device)
        p_0_embeds = self.llama_model.model.embed_tokens(p_0_tokens.input_ids).expand(batch_size, -1, -1)
        p_1_embeds = self.llama_model.model.embed_tokens(p_1_tokens.input_ids).expand(batch_size, -1, -1)
        p_2_embeds = self.llama_model.model.embed_tokens(p_2_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_embeds = torch.cat([p_0_embeds, pc_embed, p_1_embeds, scene_embed, p_2_embeds], dim=1)
        # attn_before_scene = torch.ones(batch_size, p_0_embeds.shape[1]+pc_embed.shape[1]+p_1_embeds.shape[1], dtype=torch.long).to(pc_embed.device)
        # attn_after_scene = torch.ones(batch_size, p_2_embeds.shape[1], dtype=torch.long).to(pc_embed.device)
        wrapped_atts = scene_attn[:, :1].expand(-1, wrapped_embeds.shape[1])
        # wrapped_atts = torch.cat([attn_before_scene, scene_attn, attn_after_scene], dim=1)
        return wrapped_embeds, wrapped_atts

    def encode_and_project(self, feat, attr):
        pos_emb = self.pos_embedding(attr[:, :, :3], self.input_dim).permute(0, 2, 1)
        size_color_emb = self.color_size_proj(attr[:, :, 3:])
        feat = self.input_norm(feat + size_color_emb + pos_emb)
        feat = self.scene_proj(feat)
        return feat

    def forward_stage1(self, scene_feat, scene_attr, target_id, target_captions, is_eval=False, **kwargs):
        pc_feat = torch.gather(scene_feat, 1, target_id.unsqueeze(1).unsqueeze(2).expand(-1, -1, scene_feat.shape[-1]))
        pc_attr = torch.gather(scene_attr, 1, target_id.unsqueeze(1).unsqueeze(2).expand(-1, -1, scene_attr.shape[-1]))
        pc_embed = self.encode_and_project(pc_feat, pc_attr).squeeze(1)

        target_embeds = []
        for target_caption in target_captions:
            target_tokens = self.llama_tokenizer(
                target_caption,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(pc_embed.device)
            token_mask = target_tokens["attention_mask"].unsqueeze(-1)
            target_embed = self.llama_model.model.embed_tokens(target_tokens.input_ids)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            target_embed = (target_embed * token_mask).sum(1) / token_mask.sum(1)
            target_embed = target_embed.mean(dim=0)
            target_embeds.append(target_embed)
        target_embeds = torch.stack(target_embeds, dim=0).to(pc_embed.device)
        cosine_loss = F.cosine_embedding_loss(pc_embed, target_embeds.detach(), torch.tensor([1]).to(pc_embed.device))
        l2_loss = F.mse_loss(pc_embed, target_embeds.detach())
        cosine_score = 1. - cosine_loss.detach().cpu()
        return dict(
            loss=cosine_loss,
            cosine_score=cosine_score,
            l2_dis=l2_loss.detach().cpu()
        )

    def forward_stage2(self, scene_feat, scene_attr, scene_mask, target_id, text_input, is_eval=False, **kwargs):
        obj_num = scene_feat.shape[1]
        scene_feat = self.encode_and_project(scene_feat, scene_attr)
        pc_embed = torch.gather(scene_feat, 1, target_id.unsqueeze(1).unsqueeze(2).expand(-1, -1, scene_feat.shape[-1]))
        if self.encoder_num_layers > 0:
            scene_feat = self.relation_module(scene_feat, mask=(~scene_mask.bool()).unsqueeze(1).expand(-1, obj_num, -1).unsqueeze(1))

        scene_embed = scene_feat * scene_mask.unsqueeze(-1)
        scene_attn = torch.ones(scene_embed.size()[:-1], dtype=torch.long).to(scene_embed.device)

        prompt = random.choice(self.prompt_list)
        scene_embed, scene_attn = self.prompt_wrap(pc_embed, scene_embed, scene_attn, prompt, is_eval)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in text_input]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(pc_embed.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([scene_attn.shape[0], scene_attn.shape[1]+1],
                    dtype=torch.long).to(pc_embed.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = scene_embed.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = scene_attn[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, scene_embed, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, scene_attn, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        return dict(
            loss=outputs.loss,
        )

    def forward_stage3(self, scene_feat, scene_attr, scene_mask, target_id, conversations, is_eval=False, **kwargs):
        batch_size, obj_num, _ = scene_feat.shape
        scene_feat = self.encode_and_project(scene_feat, scene_attr)
        pc_embed = torch.gather(scene_feat, 1, target_id.unsqueeze(1).unsqueeze(2).expand(-1, -1, scene_feat.shape[-1]))
        if self.encoder_num_layers > 0:
            scene_feat = self.relation_module(scene_feat, mask=(~scene_mask.bool()).unsqueeze(1).expand(-1, obj_num, -1).unsqueeze(1))

        scene_embed = scene_feat * scene_mask.unsqueeze(-1)
        # scene_attn = torch.ones(scene_embed.size()[:-1], dtype=torch.long).to(scene_embed.device)
        max_len = 0
        input_embed_list = []
        p_0_len_list, p_1_len_list = [], []
        target_list = []
        for idx, prompt in enumerate(conversations):
            tmp_scene_embed = scene_embed[idx:idx+1]
            tmp_pc_embed = pc_embed[idx:idx+1]
            p_0, p_ = prompt.split("<TargetHere>")
            p_1, p_2 = p_.split("<SceneHere>")
            p_1 = self.pc_end_token + p_1
            p_0_tokens = self.llama_tokenizer(p_0, return_tensors="pt", add_special_tokens=is_eval).to(tmp_pc_embed.device)
            p_1_tokens = self.llama_tokenizer(p_1, return_tensors="pt", add_special_tokens=False).to(tmp_pc_embed.device)
            p_2_tokens = self.llama_tokenizer(p_2, return_tensors="pt", add_special_tokens=False).to(tmp_pc_embed.device)
            p_0_embeds = self.llama_model.model.embed_tokens(p_0_tokens.input_ids)
            p_1_embeds = self.llama_model.model.embed_tokens(p_1_tokens.input_ids)
            p_2_embeds = self.llama_model.model.embed_tokens(p_2_tokens.input_ids)
            input_embeds = torch.cat([p_0_embeds, tmp_pc_embed, p_1_embeds, tmp_scene_embed, p_2_embeds], dim=1)

            sep1 = self.begin_signal + self.role[0] + ": "
            sep2 = self.begin_signal + self.role[1] + ": "
            raw_text = p_2.split(sep2)
            for _idx in range(1, len(raw_text)):
                raw_text[_idx] = sep2 + raw_text[_idx]
            answer_targets = p_2_tokens.input_ids.clone()
            system = raw_text[0].split(sep1)[0]
            system_len = self._get_text_len(system.rstrip())
            sep_len = self._get_text_len(sep1.rstrip())
            cur_len = self._get_text_len(raw_text[0].rstrip())
            answer_targets[:, :system_len] = -100
            answer_targets[:, (system_len+sep_len):cur_len] = -100
            for text in raw_text[1:-1]:
                total_len = self._get_text_len(text.rstrip())
                ans_len = self._get_text_len((text.split(sep1)[0]+sep1).rstrip())
                answer_targets[:, (cur_len+ans_len):(cur_len+total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(raw_text[-1].rstrip())
            if cur_len != answer_targets.shape[1]:
                print(f"The final length is not equal to the original prompt: {prompt}")
            assert cur_len == answer_targets.shape[1]

            max_len = max(max_len, input_embeds.shape[1])
            input_embed_list.append(input_embeds)
            p_0_len_list.append(p_0_tokens.input_ids.shape[1])
            p_1_len_list.append(p_1_tokens.input_ids.shape[1])
            target_list.append(answer_targets)

        txt_len = min(max_len + 1, self.max_txt_len + obj_num + 1)
        inputs_embeds = torch.ones([batch_size, txt_len], dtype=torch.long).to(pc_embed.device) * self.llama_tokenizer.pad_token_id
        inputs_embeds = self.llama_model.model.embed_tokens(inputs_embeds)
        attention_mask = torch.zeros([batch_size, txt_len], dtype=torch.long).to(pc_embed.device)
        targets = torch.ones([batch_size, txt_len], dtype=torch.long).to(pc_embed.device).fill_(-100)
        inputs_embeds[:, :1] = self.llama_tokenizer.bos_token_id
        for idx in range(batch_size):
            input_len = min(input_embed_list[idx].shape[1], txt_len - 1)
            inputs_embeds[idx, 1:(input_len+1)] = input_embed_list[idx][:, :input_len]
            attention_mask[idx, :(input_len+1)] = 1
            p_0_len = p_0_len_list[idx]
            p_1_len = p_1_len_list[idx]
            targets[idx, (p_0_len+p_1_len+obj_num+2):(input_len+1)] = target_list[idx][0, :(input_len-p_0_len-p_1_len-obj_num-1)]

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets
            )

        return dict(
            loss=outputs.loss
        )

    def evaluate(self, scene_feat, scene_attr, scene_mask, target_id, custom_prompt, is_eval=True, **kwargs):
        batch_size, obj_num = scene_feat.shape[:2]
        scene_feat = self.encode_and_project(scene_feat, scene_attr)
        pc_embed = torch.gather(scene_feat, 1, target_id.unsqueeze(1).unsqueeze(2).expand(-1, -1, scene_feat.shape[-1]))
        if self.encoder_num_layers > 0:
            scene_feat = self.relation_module(scene_feat, mask=(~scene_mask.bool()).unsqueeze(1).expand(-1, obj_num, -1).unsqueeze(1))

        scene_embed = scene_feat * scene_mask.unsqueeze(-1)
        scene_attn = torch.ones(scene_embed.size()[:-1], dtype=torch.long).to(scene_embed.device)

        output_texts = []
        for idx in range(batch_size):
            tmp_scene_embed, _ = self.prompt_wrap(pc_embed[idx:idx+1], scene_embed[idx:idx+1], scene_attn[idx:idx+1], custom_prompt[idx], is_eval)
            # print(scene_embed.shape[1])
            stop_words_ids = [torch.tensor([835]).to(tmp_scene_embed.device),
                              torch.tensor([2277, 29937]).to(tmp_scene_embed.device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
            outputs = self.llama_model.generate(
                inputs_embeds=tmp_scene_embed,
                max_new_tokens=min(self.max_txt_len * 2, 512),
                stopping_criteria=stopping_criteria,
                num_beams=1,
                do_sample=True,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.0,
                length_penalty=1,
                temperature=1.0,
            )
            output_token = outputs[0]
            if output_token[0] == 0:  # the model might output an unknown token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]
            output_texts.append(output_text)

        return output_texts

    def forward(self, **kwargs):
        if "target_captions" in kwargs:
            return self.forward_stage1(**kwargs)
        if "text_input" in kwargs:
            return self.forward_stage2(**kwargs)
        if "conversations" in kwargs:
            return self.forward_stage3(**kwargs)
        if "custom_prompt" in kwargs:
            return self.evaluate(**kwargs)
        return None

    def _get_text_len(self, text):
        return self.llama_tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @property
    def device(self):
        return list(self.parameters())[0].device
