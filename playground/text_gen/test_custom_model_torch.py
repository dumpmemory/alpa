from collections import namedtuple
import time
import torch
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel
from transformers.generation_utils import GenerationMixin, ModelOutput, dataclass


@dataclass
class InferenceFuncOutput(ModelOutput):
    logits: any = None
    past_key_values: any = None


class WrappedInferenceFunc(GenerationMixin):
    def __init__(self, inference_func, config):
        self.inference_func = inference_func 
        self.config = config
        self.main_input_name = "input_ids"

    def forward(self, attention_mask):
        raise NotImplementedError()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for input_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
        }

    def __call__(self,
                 input_ids,
                 past_key_values = None,
                 output_attentions = None,
                 output_hidden_states = None,
                 return_dict = None):
        for i in range(input_ids.shape[1]):
            ret = self.inference_func(input_ids[:,i:i+1],
                                      past_key_values)
            past_key_values = ret.past_key_values
        return ret


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

if "gpt" in model_name:
    raw_model = GPT2LMHeadModel.from_pretrained(model_name)

    def inference_func(input_ids, past_key_values):
        assert input_ids.shape[1] == 1, f"{input_ids.shape}"
        out = raw_model(input_ids=input_ids,
                        past_key_values=past_key_values)
        return InferenceFuncOutput(out.logits, out.past_key_values)

elif "opt" in model_name:
    raw_model = OPTForCausalLM.from_pretrained(model_name)

    def inference_func(input_ids, past_key_values):
        assert input_ids.shape[1] == 1, f"{input_ids.shape}"
        if past_key_values is None:
            attention_mask = None
        else:
            past_length = past_key_values[0][0].shape[2]
            attention_mask = torch.ones((input_ids.shape[0], past_length+1))
        out = raw_model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
        return InferenceFuncOutput(out.logits, out.past_key_values)
 
model = WrappedInferenceFunc(inference_func, raw_model.config)

torch.manual_seed(9)
prompt = "Computer science is the study of computation and"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids=input_ids, max_length=20, do_sample=True)
generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(input_ids)
print(generated_string)
