from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from datasets import load_dataset
from unsloth import get_chat_template
from transformers import TextStreamer
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-3-4b-pt",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)



model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,                  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               # We support rank stabilized LoRA
    loftq_config = None,               # And LoftQ
    target_modules = "all-linear",    # Optional now! Can specify a list if needed
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
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

# FastVisionModel.for_inference(model)  # Enable for inference!

# image = eval_dataset[0]["yolo_detection_image"]
# instruction = eval_dataset[0]["question"]

# messages = [
#     {
#         "role": "user",
#         "content": [{"type": "image"}, {"type": "text", "text": instruction}],
#     }
# ]
# input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
# inputs = processor(
#     image,
#     input_text,
#     add_special_tokens=False,
#     return_tensors="pt",
# ).to("cuda")



# text_streamer = TextStreamer(processor, skip_prompt=True)
# result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
#                         use_cache=True, temperature = 1.0, top_p = 0.95, top_k = 64)





FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_train_dataset,
    eval_dataset=converted_eval_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,

        # use reentrant checkpointing
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        max_grad_norm = 0.3,              # max gradient norm based on QLoRA paper
        warmup_ratio = 0.03,
        max_steps = 10,
        #num_train_epochs = 2,          # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        save_strategy="steps",
        save_steps=5,
        eval_strategy="steps",
        eval_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False, 
        # save_total_limit = 1, # save only the best model 
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",             # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_seq_length = 2048,
    )
)

trainer_stats = trainer.train()

model.save_pretrained("outputs/best_gemma_model")  # Local saving
processor.save_pretrained("outputs/best_gemma_model")