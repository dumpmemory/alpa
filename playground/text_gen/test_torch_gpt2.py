import time
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "Computer science is the study of computation, automation, and information."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids=input_ids, max_length=50, do_sample=True)
generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(input_ids)
print(generated_string)

import torch
costs = []
for i in range(10):
    torch.cuda.synchronize()
    tic = time.time()
    generated_ids = model.generate(input_ids=input_ids, max_length=50, do_sample=True)
    torch.cuda.synchronize()
    costs.append(time.time() - tic)
print(costs)
