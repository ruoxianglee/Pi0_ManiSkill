from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "google/paligemma-3b-mix-224"
device = "cuda:0"
dtype = torch.bfloat16

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("place_food.png")
if image.mode != "RGB":
    image = image.convert("RGB")

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    revision="bfloat16",
).eval().to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Instruct the model to create a caption in Spanish
INSTRUCTION = "place food in the skillet"
prompt = f"What action should the two robotic arms take to {INSTRUCTION}? Please take task planning."
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    for i in range(100):
        start_obs_enc = torch.cuda.Event(enable_timing=True)
        end_obs_enc = torch.cuda.Event(enable_timing=True)
        start_obs_enc.record()
        
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        
        end_obs_enc.record()        
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        obs_enc_time = start_obs_enc.elapsed_time(end_obs_enc)
        
        print(decoded)
        print(f"Observation encoding time: {obs_enc_time:.2f} ms")

