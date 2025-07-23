from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from datasets import load_dataset
from unsloth import get_chat_template
from transformers import TextStreamer
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

model, processor = FastVisionModel.from_pretrained(
    "outputs/best_gemma_model",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)





train_dataset = load_dataset("rfeiglew/ThermalRefl", split = "train")
eval_dataset = load_dataset("rfeiglew/ThermalRefl_eval_metrics", split = "train")

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : sample["question"]},
            {"type" : "image", "image" : sample["yolo_detection_image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["answer"]} ]
        },
    ]
    return { "messages" : conversation }
pass

converted_train_dataset = [convert_to_conversation(sample) for sample in train_dataset]
converted_eval_dataset = [convert_to_conversation(sample) for sample in eval_dataset]

# Take the Gemma 3 chat template
processor = get_chat_template(
    processor,
    "gemma-3"
)

FastVisionModel.for_inference(model)  # Enable for inference!

image = eval_dataset[0]["yolo_detection_image"]
instruction = eval_dataset[0]["question"]

messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
    }
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")



text_streamer = TextStreamer(processor, skip_prompt=True)
result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                        use_cache=True, temperature = 1.0, top_p = 0.95, top_k = 64)


