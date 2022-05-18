import time
from transformers import GPT2Tokenizer, FlaxGPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")

prompt = "Computer science is the study of computation, automation, and information."
input_ids = tokenizer(prompt, return_tensors="np").input_ids
generated_ids = model.generate(input_ids=input_ids, max_length=50, do_sample=True)
generated_string = tokenizer.batch_decode(generated_ids.sequences, skip_special_tokens=True)

print(input_ids)
print(generated_string)

costs = []
for i in range(10):
    tic = time.time()
    generated_ids = model.generate(input_ids=input_ids, max_length=50, do_sample=True)
    costs.append(time.time() - tic)
print(costs)
