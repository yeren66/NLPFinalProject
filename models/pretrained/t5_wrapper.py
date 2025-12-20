import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5Wrapper(nn.Module):
    def __init__(self, model_name='t5-small', device='cuda'):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = device
        
    def forward(self, src_ids, src_mask, tgt_ids, tgt_mask=None):
        # T5 handles loss calculation internally if labels are provided
        # tgt_ids usually need to be shifted? T5 handles it if we pass labels.
        # But for consistency, let's see. 
        # T5 forward: inputs_embeds/input_ids, attention_mask, decoder_inputs_ids, labels
         
        outputs = self.model(
            input_ids=src_ids,
            attention_mask=src_mask,
            labels=tgt_ids
        )
        return outputs.loss, outputs.logits
        
    def generate(self, src_text):
        self.model.eval()
        input_ids = self.tokenizer(src_text, return_tensors="pt", padding=True).input_ids.to(self.device)
        outputs = self.model.generate(input_ids)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
