import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json


def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

model = Qwen2VLForConditionalGeneration.from_pretrained("./Qwen/Qwen2-VL-2B-Instruct/", device_map="auto",
                                                        torch_dtype=torch.bfloat16, trust_remote_code=True, )
processor = AutoProcessor.from_pretrained("./Qwen/Qwen2-VL-2B-Instruct")


val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取测试模型
val_peft_model = PeftModel.from_pretrained(model, model_id="./output/Qwen2-VL-2B/checkpoint-2", config=val_config)

# 读取测试数据
with open("data_vl_test.json", "r") as f:
    test_dataset = json.load(f)

test_image_list = []
for item in test_dataset:
    # input_image_prompt = item["conversations"][0]["value"]
    # 去掉前后的<|vision_start|>和<|vision_end|>
    # origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

    input_content = item["images"]
    output_content = item["report"]

    messages = [
        {
            "role": "user",
            "content": [

            ],
        }
    ]

    for index, file_path in enumerate(input_content):
        file_path = file_path.replace("\\","/")
        # print(file_path)
        messages[0]["content"].append({
                "type": "image",
                "image": f"{file_path}",
                "resized_height": 140,
                "resized_width": 140,
            })

    messages[0]["content"].append(
        {"type": "text", "text": "请基于超声图像生成规范的超声检查报告。描述内容包括器官的位置、大小、形态完整描述，组织的回声特征详细描述，血流信号情况，病变部位的精确定位和测量，并结合所见给出明确的诊断建议"}
    )



    response = predict(messages, val_peft_model)
    messages.append({"role": "assistant", "content": f"{response}"})
    output_content = output_content.replace("\n","")
    print(output_content)
    print(messages[-1]["content"])
    file_path = './example.txt'  # 指定文件路径
    with open(file_path, 'a', encoding='utf-8') as file:  # 使用with语句确保文件在操作完成后正确关闭
        file.write("原始数据：\n\n")
        file.write(output_content)
        file.write("\n\n")
        file.write("模型输出：\n\n")
        file.write(messages[-1]["content"])
        file.write("\n\n")
    # test_image_list.append(swanlab.Image(origin_image_path, caption=response))

# swanlab.log({"Prediction": test_image_list})