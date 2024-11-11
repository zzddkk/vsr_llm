import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import hydra
import transformers
import os
import torch.nn.init as init
from typing import Optional
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pad_sequence
# from huggingface_hub import login
# # login()
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from utils import make_non_pad_mask
from transformers import GemmaForCausalLM,GemmaTokenizer,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training

class ModelModule():
    def __init__(self, cfg):
        self.cfg = cfg

    def _step(self,batch,mode):
        if mode == "train":
            outputs = self.model(batch["video"],batch["target"],batch["input_lengths"])
            return outputs.loss
        if mode == "test":
            outputs = self.model.generate(batch["video"],batch["input_lengths"])   
            return outputs  
        if mode == "val":
            outputs = self.model(batch["video"],batch["target"],batch["input_lengths"])
            return outputs.loss

    def training_step(self,batch):
        return self._step(batch,"train")

    def test_step(self,batch):
        return self._step(batch,"test")

    def val_step(self,batch):
        return self._step(batch,"val")
    
    def build_model(self,cfg):

        """
        build the model
        """
        cfg=cfg.model
        # quantization or not
        # the bnb_4bit_compute_dtype set torch.bfloat16/troch.float32 the output will imporve obviously
        if cfg.decoder.quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            tokenizer = GemmaTokenizer.from_pretrained("google/gemma-1.1-2b-it")
            model = GemmaForCausalLM.from_pretrained("google/gemma-1.1-2b-it",quantization_config=bnb_config)
            model.gradient_checkpointing_enable()
            model=prepare_model_for_kbit_training(model)
        else:
            tokenizer =GemmaTokenizer.from_pretrained("google/gemma-1.1-2b-it")
            model = GemmaForCausalLM.from_pretrained("google/gemma-1.1-2b-it")
        if cfg.decoder.Lora:
            # the lora config
            config = LoraConfig(
                    r=cfg.decoder.lora_r,
                    lora_alpha=cfg.decoder.lora_alpha,
                    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=cfg.decoder.lora_dropout,
                    bias=cfg.decoder.lora_bias,
                    task_type=cfg.decoder.lora_task_type,
                )
            model=get_peft_model(
                model,
                peft_config=config,
            )
            # print the trainable parameters
            model.print_trainable_parameters()
            model=model.model
        else:
            for name, param in model.named_parameters():
                param.requires_grad = False
        self.model=VSR_LLM(tokenizer,model,cfg)
        self.tokenizer=tokenizer
        return self.model


class VSR_LLM(nn.Module):

    """
    A video-only speech transcription model based on the Transformer architecture.
    Architecture: A stack of 12 Transformer encoder layers,
                  first 6 form the Encoder and the last 6 form the Decoder.
                  mlp + LLM(google/gemma-2-2b-it)
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Input: 511.1-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """
    def __init__(self,tokenizer,model,cfg):
        super(VSR_LLM, self).__init__()
        self.cfg=cfg
        self.tokenizer=tokenizer
        self.tokenizer.padding_side="right"
        self.tokenizer.add_tokens(["STT"])
        self.model=model
        self.prompt=cfg.decoder.prompt
        # the visual encoder layers: 3D + resnet + comformer encoder
        self.encoder = Encoder(
            attention_dim=cfg.encoder.adim,
            attention_heads=cfg.encoder.aheads,
            linear_units=cfg.encoder.eunits,
            num_blocks=cfg.encoder.elayers,
            input_layer=cfg.encoder.transformer_input_layer,
            dropout_rate=cfg.encoder.dropout_rate,
            positional_dropout_rate=cfg.encoder.dropout_rate,
            attention_dropout_rate=cfg.encoder.transformer_attn_dropout_rate,
            encoder_attn_layer_type=cfg.encoder.transformer_encoder_attn_layer_type,
            macaron_style=cfg.encoder.macaron_style,
            use_cnn_module=cfg.encoder.use_cnn_module,
            cnn_module_kernel=cfg.encoder.cnn_module_kernel,
            zero_triu=getattr(cfg.encoder, "zero_triu", False),
            a_upsample_ratio=cfg.encoder.a_upsample_ratio,
            relu_type=getattr(cfg.encoder, "relu_type", "swish"),
        )

        # the projector to project the visual features to the embedding size
        
        self.projector=nn.Linear(self.cfg.encoder.adim, self.cfg.decoder.embedding_size)
        # get the id of the special token id
        self.bos_token_id=self.tokenizer.bos_token_id
        self.eos_token_id=self.tokenizer.eos_token_id
        assert self.eos_token_id,"the end token id is not found,can not successfully get the end token id"
        assert self.bos_token_id,"the end token id is not found,can not successfully get the end token id"
        # tokenizer and embedding the prompt
        prompt_token = self.tokenizer(self.prompt, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        str_model_token = self.tokenizer("model\n", add_special_tokens=False, return_tensors="pt").to(self.model.device)
        self.prompt_embed_token = self.model.model.embed_tokens(prompt_token.input_ids).to(self.model.device)
        self.str_embed_tokens = self.model.model.embed_tokens(str_model_token.input_ids).to(self.model.device)
        # embed the special tokens
        self.bos_embed_token=self.model.model.embed_tokens(torch.tensor(self.eos_token_id).to(self.model.device))
        self.eos_embed_token=self.model.model.embed_tokens(torch.tensor(self.eos_token_id).to(self.model.device))
    
    def add_eos(self,batch):
        res = []
        for t in batch:
            t = t + "<eos>"
            res.append(t)
        return res
        
    def get_the_input_embeds(self,batch,input_lengths,targetBatch:Optional[torch.Tensor]=None):

        """
        the function will return the input_embeds to caculate the loss
        CTC_loss ,crossentropy,seq2seq_loss
        """

        B,T,D=batch.shape

        bos_embed_tokens=self.bos_embed_token.repeat(B,1,1)
        eos_embed_tokens=self.eos_embed_token.repeat(B,1,1)
        prompt_embed_tokens = self.prompt_embed_token.repeat(B,1,1)
        str_embed_tokens = self.str_embed_tokens.repeat(B,1,1)
        if targetBatch:
            transcripts = self.add_eos(targetBatch)
            inputs_embeds = []
            to_regress_tokens = self.tokenizer(transcripts,return_tensors="pt",add_special_tokens=False,padding=True).to(self.model.device)
            # the transcript txt + <eos>
            to_regress_embed_tokens = self.model.model.embed_tokens(to_regress_tokens.input_ids)
            transcript_mask = to_regress_tokens.attention_mask
            target_length = transcript_mask.sum(-1)
            # the input_embeds
            for i in range(B):
                video = batch[i][:input_lengths[i]]
                transcript = to_regress_embed_tokens[i][:target_length[i]]
                inputs_embeds.append(torch.cat([bos_embed_tokens[i],prompt_embed_tokens[i],video,str_embed_tokens[i],to_regress_embed_tokens[i],eos_embed_tokens[i]],dim=0))
            inputs_embeds = pad_sequence(inputs_embeds,batch_first=True,padding_value=0,padding_side="left")
            target_ids = torch.zeros(inputs_embeds.shape[0],inputs_embeds.shape[1],dtype=torch.long).fill_(-100)
            transcript_ids = to_regress_tokens.input_ids
            for i in range(B):
                target_ids[i][-target_length[i]:] = transcript_ids[i][:target_length[i]]
            return inputs_embeds,target_ids
        else:
            inputs_embeds = []
            for i in range(B):
                video = batch[i][:input_lengths[i]]
                inputs_embeds.append(torch.cat([bos_embed_tokens[i],prompt_embed_tokens[i],video,str_embed_tokens[i]],dim=0))
            inputs_embeds = pad_sequence(inputs_embeds,batch_first=True,padding_value=0,padding_side="left")
            return inputs_embeds
                    

    def forward(self, inputBatch,targetBatch,input_lengths):

        """
        input: batch of visual features
        output: 
            type:   dict 
            framework:  the dict contains the logit,loss and etc.
        the format of the input is as follows:
        <bos><start_of_turn>user
        You are a helpful AI assistant for recognizing the input speech in English.Input<end_of_turn>
        <start_of_turn>model 

        OR:
        <bos>user
        You are a helpful AI assistant for recognizing the input speech in English.Input<end_of_turn>
        model

        the format of the targets is as follows:
        [-100]*len(input_embeds[1]-to_regress_embeds.shape[1])+[target_ids]
        we will use -100 to fill the part of prompt and video features beacuse we do not need to calculate the loss of them
        ans crossentropy loss will ignore the -100
        """
        inputBatch = inputBatch.to(self.model.device)
        padding_mask = make_non_pad_mask(input_lengths).to(inputBatch.device).unsqueeze(-2)
        batch,_ = self.encoder(inputBatch,padding_mask)
        batch = self.projector(batch)
        """
        From the huggingface transformers library:
        model predict the next token
        if labels is not None:
        use the provided labels as the next token to supervise the model
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()
            loss = None
            if labels is not None:
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.float()
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
        """
        inputs_embeds,target_ids = self.get_the_input_embeds(batch,input_lengths,targetBatch)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            return_dict=True,
            labels=target_ids,
        )
                        
        return outputs


    @torch.no_grad()
    def generate(self,inputBatch,input_lengths):

        """
        the function will generate the text logits if set output_logits=True
        """
        inputBatch = inputBatch.to(self.model.device)
        padding_mask = make_non_pad_mask(input_lengths).unsqueeze(-2)
        batch , _ = self.encoder(inputBatch,padding_mask)
        batch = self.projector(batch)
        inputs_embeds=self.get_the_input_embeds(batch,input_lengths,None)    
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            top_p=self.cfg.decoder.top_p,
            repetition_penalty=self.cfg.decoder.repetition_penalty,
            length_penalty=self.cfg.decoder.length_penalty,
            num_beams=self.cfg.decoder.num_beams,
            return_dict_in_generate=True,
            # output_logits=True,
            max_new_tokens=self.cfg.decoder.max_new_tokens,
        )
        return outputs

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
@hydra.main(config_path=os.path.join(parent_dir,"conf"), config_name="configs")
def test(cfg):
    """
    test function
    """
    modelmodule = ModelModule(cfg)
    model = modelmodule.build_model(cfg)
    model = model.to("cuda")
    from transform import VideoTransform
    from vsr_llm_datamodule import DataModule
    datamodel = DataModule(cfg)
    val_dataloader = datamodel.val_dataloader(batch_size=1)
    for batch in val_dataloader:
        inputBatch=batch["video"]
        targetBatch=batch["target"]
        input_lengths=batch["input_lengths"]
        inputBatch = inputBatch.to("cuda")
        input_lengths = input_lengths.to("cuda")
        outputs = model(inputBatch,targetBatch,input_lengths)
        for k in outputs.keys():
            print(k)
        break
    # for n,p in model.named_parameters():
    #     print(n,p.requires_grad)
    # print(model)
if __name__ == "__main__":
    test()
