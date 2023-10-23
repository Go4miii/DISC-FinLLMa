from abc import ABCMeta, abstractmethod
import re

from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import torch
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', required=True, help='which cuda device you want to inference')
parser.add_argument('--model', type=str, help='自定义LLM名称')
parser.add_argument('--lora_path', default='', type=str, help='自定义LLM名称')
args = parser.parse_args()

model = args.model
lora_path = None if args.lora_path == '' else args.lora_path

DEVICE = args.device


class DISCFINLLMBase(metaclass=ABCMeta):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        # 模型需要接收提示prompt，使用模型生成回复
        raise NotImplementedError


class DISCVFINLLMChatGLM26B(DISCFINLLMBase):
    def __init__(self, lora_path: str = None):
        model_name_or_path = "THUDM/chatglm2-6b"
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype).to(DEVICE)  # .half().cuda()
        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = self.model.eval()

    def generate(self, prompt: str) -> str:
        answer, history = self.model.chat(self.tokenizer, prompt, history=[])
        return answer


class DISCVFINLLMChatGLM6B(DISCFINLLMBase):
    def __init__(self, lora_path: str = None):
        model_name_or_path = "THUDM/ChatGLM-6B"
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype).to(DEVICE)  # .half().cuda()
        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = self.model.eval()

    def generate(self, prompt: str) -> str:
        answer, history = self.model.chat(self.tokenizer, prompt, history=[])
        return answer


class DISCVFINLLMBaichuan13BBase(DISCFINLLMBase):
    def __init__(self, lora_path: str = None):
        model_name_or_path = "baichuan-inc/Baichuan-13B-Base"
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                          torch_dtype=torch.float16,
                                                          trust_remote_code=True).to(DEVICE)
        self.model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan-13B-Base")

        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)

    def generate(self, prompt: str) -> str:
        template = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            "Human: {}\nAssistant: "
        )

        inputs = self.tokenizer([template.format(prompt)], return_tensors="pt")
        inputs = inputs.to(DEVICE)
        generate_ids = model.generate(**inputs, max_new_tokens=256)

        return generate_ids


class DISCVFINLLMBaichuan13BChat(DISCFINLLMBase):
    def __init__(self, lora_path: str = None):
        model_name_or_path = "baichuan-inc/Baichuan-13B-Chat"
        # model_name_or_path='/root/.cache/huggingface/hub/models--baichuan-inc--Baichuan-13B-Chat/snapshots/e580bc803f3f4f6b42ddccd0730739c057c7b54c'
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                          torch_dtype=torch.float16,
                                                          trust_remote_code=True).to(DEVICE)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
            print('lora加载完！')

    def generate(self, prompt: str) -> str:
        messages = []
        messages.append({"role": "user", "content": prompt})
        response = self.model.chat(self.tokenizer, messages)

        return response


class DISCVFINLLMBaichuan7B(DISCFINLLMBase):
    def __init__(self, lora_path: str = None):
        model_name_or_path = "baichuan-inc/Baichuan-7B"
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).half()
        self.model = self.model.to(DEVICE)

        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)

    def generate(self, prompt: str) -> str:
        template = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            "Human: {}\nAssistant: "
        )

        inputs = self.tokenizer(template.format(prompt), return_tensors='pt')
        inputs = inputs.to(DEVICE)
        pred = self.model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
        answer = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        print(answer)
        pattern = answer.split('Assistant: ', 1)

        assistant_text = pattern[-1]
        print(assistant_text)
        return assistant_text


class DISCVFINLLMBloomz7B(DISCFINLLMBase):
    def __init__(self, lora_path: str = None):
        model_name_or_path = "bigscience/bloomz-7b1-mt"
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).half().to(DEVICE)

        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)

    def generate(self, prompt: str) -> str:
        # template = (
        #     "A chat between a curious user and an artificial intelligence assistant. "
        #     "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        #     "Human: {}\nAssistant: "
        # )
        # inputs = self.tokenizer.encode(template.format(prompt), return_tensors="pt").to(DEVICE)
        # outputs = self.model.generate(inputs)
        # # outputs=self.model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
        # # answer = self.tokenizer.decode(outputs[0]).replace(prompt, '')
        # answer=self.tokenizer.decode(outputs[0])
        # # start_index = answer.find("Assistant:")
        # # end_index = answer.find("</s>")
        # # # 提取内容
        # # if start_index != -1 and end_index != -1:
        # #     extracted_text = answer[start_index + len("Assistant:"):end_index]
        # #     print(extracted_text)
        # # else:
        # #     print("未找到合适的内容.")
        # pattern = r'Assistant: (.+?)(?:</s>|$)'

        # # 使用findall函数提取匹配的内容
        # matches = re.findall(pattern, answer)
        # # 输出结果
        # if matches!=[]:
        #     assistant_text = matches[0]
        # else:
        #     assistant_text='无'

        template = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            "Human: {}\nAssistant: "
        )
        inputs = self.tokenizer.encode_plus(template.format(prompt), return_tensors='pt')
        outputs = self.model.generate(**inputs.to(DEVICE), max_new_tokens=128, repetition_penalty=1.1)
        # answer = self.tokenizer.decode(outputs[0]).replace(prompt, '')
        answer = self.tokenizer.decode(outputs[0])
        # start_index = answer.find("Assistant:")
        # end_index = answer.find("</s>")
        # # 提取内容
        # if start_index != -1 and end_index != -1:
        #     extracted_text = answer[start_index + len("Assistant:"):end_index]
        #     print(extracted_text)
        # else:
        #     print("未找到合适的内容.")
        pattern = r'Assistant: (.+?)(?:</s>|$)'

        # 使用findall函数提取匹配的内容
        matches = re.findall(pattern, answer)
        # 输出结果
        if matches != []:
            assistant_text = matches[0]
        else:
            assistant_text = '无'

        return assistant_text


class FinGPTv3:
    def __init__(self):
        model_name_or_path = "THUDM/chatglm2-6b"
        peft_model = "oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT"
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(DEVICE)
        self.model = PeftModel.from_pretrained(self.model, peft_model)

    def generate(self, prompt: str) -> str:
        tokens = self.tokenizer(prompt, return_tensors='pt', padding=True, max_length=512)
        res = self.model.generate(**tokens.to(DEVICE), max_length=512)
        # res_sentences = [tokenizer.decode(i) for i in res]
        res_sentences = self.tokenizer.decode(res[0])
        # print(res_sentences)
        answer = res_sentences.replace(prompt, '').strip()
        # out_text = [o.split("答案：")[-1] for o in res_sentences]
        return answer


if __name__ == '__main__':
    """
    写的一些测试函数
    """
    # 原始模型
    llm = DISCVFINLLMChatGLM26B()
    llm.generate('你好')

    # LORA模型
    llm = DISCVFINLLMChatGLM26B(
        lora_path="/remote-home/qswang/chatglm2/zero_nlp/chatglm_v2_6b_lora/output/fin_few_v1/checkpoint-1539"
    )
    llm.generate('你好')

