from transformers import CLIPVisionModel, T5ForConditionalGeneration, T5ForQuestionAnswering
from torch import nn
from peft import TaskType, LoraConfig, get_peft_model
import pdb
import torch
from einops import rearrange
import inspect


class Clip_FlanT5(nn.Module):
    def __init__(self, args):
        super(Clip_FlanT5, self).__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(args.vision_model_name_or_path, cache_dir=args.cache_dir)
        self.text_decoder = T5ForConditionalGeneration.from_pretrained(args.text_model_name_or_path, cache_dir=args.cache_dir)
        del self.text_decoder.encoder

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.vision_projector = nn.Linear(self.vision_encoder.config.hidden_size, self.text_decoder.config.d_model)

        # apply lora to text_decoder
        if args.use_lora:
            if self.text_decoder.decoder.config.is_gated_act:
                peft_config = LoraConfig(
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    r=args.lora_r,
                    bias="none",
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    inference_mode=False,
                    target_modules=[
                        "q",
                        "SelfAttention.k",
                        "EncDecAttention.k",
                        "v",
                        "o",
                        "wi_0",
                        "wi_1",
                        "wo",
                    ]
                )
            else:
                peft_config = LoraConfig(
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    r=args.lora_r,
                    bias="none",
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    inference_mode=False,
                    target_modules=[
                        "q",
                        "SelfAttention.k",
                        "EncDecAttention.k",
                        "v",
                        "o",
                        "wi",
                        "wo",
                    ]
                )
            
            self.text_decoder = get_peft_model(self.text_decoder, peft_config)
            self.text_decoder.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, pixel_values):
        with torch.no_grad():
            visual_features = self.vision_encoder(pixel_values=pixel_values)
        visual_features.last_hidden_state = self.vision_projector(visual_features.last_hidden_state[:, 1:])

        input_ids[input_ids == 0] = -100

        outputs = self.text_decoder(decoder_attention_mask=attention_mask, encoder_outputs=visual_features, return_dict=True, labels=input_ids)
        return outputs.loss
    
    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values):
        visual_features = self.vision_encoder(pixel_values=pixel_values)
        visual_features.last_hidden_state = self.vision_projector(visual_features.last_hidden_state[:, 1:])
        outputs = self.text_decoder.generate(decoder_attention_mask=attention_mask, encoder_outputs=visual_features, max_new_tokens=30, num_beams=4, repetition_penalty=1.5, length_penalty=1.0, early_stopping=True, use_cache=True)

        return outputs
