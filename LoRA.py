import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    TrainingArguments,
    Trainer
)
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import os
import logging
from torch.utils.data import Dataset as TorchDataset

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenVLDataset(TorchDataset):
    """
    自定义数据集类，用于处理Qwen2-VL的图像-文本对
    """

    def __init__(self, data_path, base_path, processor):
        self.processor = processor

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.base_path = base_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建图像路径
        image_path = os.path.join(self.base_path, item["image"])

        # 构建对话消息
        conversations = item["conversations"]
        messages = []

        for conv in conversations:
            if conv["from"] == "human":
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": conv["value"]}
                    ]
                })
            elif conv["from"] == "gpt":
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": conv["value"]}
                    ]
                })

        # 应用聊天模板
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # 处理图像和文本输入
        image_inputs, _ = process_vision_info(messages)

        # 处理输入
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        # 返回处理后的输入
        result = {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": inputs["input_ids"].squeeze()
        }
        
        # 添加所有图像相关的张量（Qwen2-VL模型需要）
        for key in ["pixel_values", "image_grid_thw"]:
            if key in inputs:
                result[key] = inputs[key].squeeze()
        
        return result

# 自定义Trainer类，适配最新transformers库的参数规范
class QwenVLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写compute_loss，适配最新Trainer的参数规范
        """
        # 移除labels并计算损失
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # 计算因果语言模型损失
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1)
        )

        return (loss, outputs) if return_outputs else loss

def setup_lora_model(model_path):
    """
    设置LoRA模型
    """
    logger.info("正在加载基础模型...")

    # 使用BitsAndBytesConfig替代废弃的load_in_8bit参数
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    except ImportError:
        quantization_config = None

    # 加载基础模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config if quantization_config else None,
        low_cpu_mem_usage=True
    )

    # 准备模型进行量化训练
    model = prepare_model_for_kbit_training(model)

    # 配置LoRA参数
    lora_config = LoraConfig(
        r=8,  # LoRA秩
        lora_alpha=16,  # LoRA缩放因子
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head"
        ],  # 针对视觉和语言模块
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 应用LoRA配置
    model = get_peft_model(model, lora_config)

    logger.info("LoRA配置完成")
    model.print_trainable_parameters()

    return model


def main():
    # 模型和数据路径
    model_path = "/mnt/f/python_work/training/model/Qwen/Qwen2-VL-2B-Instruct"
    training_data_path = "/mnt/f/python_work/training/Qwen3-VL/qwen-vl-finetune/demo/single_images.json"
    output_dir = "/mnt/f/python_work/training/lora_checkpoints"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载处理器
    logger.info("正在加载处理器...")
    processor = Qwen2VLProcessor.from_pretrained(model_path)

    # 设置LoRA模型
    model = setup_lora_model(model_path)

    # 创建数据集
    logger.info("正在准备训练数据...")
    base_path = "/mnt/f/python_work/training/Qwen3-VL/qwen-vl-finetune/"
    train_dataset = QwenVLDataset(training_data_path, base_path, processor)

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=500,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        remove_unused_columns=False,
        push_to_hub=False,
        save_total_limit=2,
        load_best_model_at_end=False,
        dataloader_pin_memory=False,
        fp16=True,
        report_to=None,
        disable_tqdm=False,
    )

    # 创建自定义训练器
    trainer = QwenVLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 开始训练
    logger.info("开始LoRA微调...")
    trainer.train()

    # 保存LoRA权重
    logger.info("保存LoRA微调模型...")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    logger.info(f"LoRA微调完成，模型已保存到: {output_dir}")


if __name__ == "__main__":
    main()