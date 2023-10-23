<div align="center">

ZH | [EN](./README-en.md)

<h1>DISC-FinLLM</h1>
  
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/ShengbinYue/DISC-LawLLM)
[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](./LICENSE)

[Demo](https://law.fudan-disc.com) | [技术报告](https://arxiv.org/abs/2309.11325)

</div>

DISC-FinLLM 是一个专门针对金融场景下为用户提供专业、智能、全面的**金融咨询服务**的金融领域大模型，由[复旦大学数据智能与社会计算实验室 (Fudan-DISC)](http://fudan-disc.com) 开发并开源。

我们将在该项目中开源如下资源：
<!-- * [DISC-Fin-SFT 数据集](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT)（不包括法律问答部分） -->
* [DISC-FinLLM 模型权重](https://huggingface.co/ShengbinYue/DISC-LawLLM)
* [DISC-Fin-Eval Benchmark](https://huggingface.co/ShengbinYue/DISC-LawLLM)

您可以通过访问这个[链接](https://fin.fudan-disc.com)来在线体验我们的 DISC-FinLLM。

<!-- - [模型微调](#模型微调) -->

## 目录

- [概述](#概述)
- [推理和部署](#推理和部署)
- [DISC-Fin-Eval Benchmark](#disc-fin-eval-benchmark)
- [致谢](#致谢)
- [声明](#声明)
- [引用](#引用)
- [协议](#协议)

## 概述

![Image](./images/model_zh.png)

<p></p>

DISC-FinLLM是一个金融领域的大语言模型，是由面向不同金融场景的4个模组：金融咨询、金融文本分析、金融计算、金融知识检索问答构成的多专家智慧金融系统。这些模组分别在金融NLP任务、人类试题、资料分析和时事分析等四个评测中展现出明显优势，证明了DISC-FinLLM能为广泛的金融领域提供强有力的支持。DISC-FinLLM能在不同应用场景下提供帮助，可用于实现不同的功能：

<!-- DISC-LawLLM 是一个具有法律推理和知识检索能力的智能法律系统，它面向不同群体，能在不同应用场景下提供帮助，主要有以下几个特点： -->

* **金融咨询：** 该模组可以在中国金融语境下，与用户展开关于金融话题的多轮对话，或是为用户解释金融专业的相关知识，是由数据集中的金融咨询指令部分训练而来。
* **金融文本分析：** 该模组可以帮助用户在金融文本上完成的信息抽取、情感分析、文本分类、文本生成等NLP任务，是由数据集中的金融任务指令部分训练而来。
* **金融计算：** 该模组可以帮助用户完成与数学计算相关的任务，除了利率、增长率等基本计算，它还支持统计分析和包括Black-Scholes期权定价模型、EDF预期违约概率模型在内的金融模型计算。这一模组是由数据集中的金融计算指令部分训练而来。
* **金融知识检索问答：** 该模组可以基于金融新闻、研报和相关政策文件为用户提供投资建议、时事分析、政策解读。它是由数据集中的检索增强指令部分训练而来。


<!-- 除此之外，我们的研究过程还包括了如下贡献：

* **高质量的训练数据集和普遍有效的训练范式**
* **完备的法律模型测评框架和测评数据集** -->



### 模型效果演示

#### 金融咨询

![consult_demo](./images/example_consult.gif)

#### 金融文本分析

![document_demo](./images/example_task.gif)

#### 金融计算

![tool_demo](./images/example_tool.gif)

#### 金融知识检索问答

![exam_ref_demo](./images/example_retrieval.gif)

<!-- #### 法条检索

![law_ref_demo](./images/example_law_ref.gif)

#### 带检索的法律咨询

![consult_ref_demo](./images/example_consult_ref.gif) -->

### DISC-Fin-SFT 数据集
DISC-FinLLM是基于我们构建的高质量金融数据集DISC-Fin-SFT在通用领域中文大模型Baichuan-13B-Chat上进行LoRA指令微调得到的金融大模型。DISC-Fin-SFT总共包含约25万条数据，分为四个子数据集，它们分别是金融咨询指令、金融任务指令、金融计算指令、检索增强指令。

![Image](./images/data_zh.png)

#### 金融咨询指令
金融咨询指令数据来源于两部分：
（1） 金融问答数据集。我们首先选择的金融问答数据集是FiQA，由于这是一个英文数据集且回答的答案质量存在一定的不足，因此我们将FiQA中的所有问题翻译成中文，并使用ChatGPT重新生成在中国背景下此问题的答案。除此之外，我们还根据200多个金融名词，针对每个名词让ChatGPT生成对应的问题，并要求在中国背景下回答这些问题。
（2）经管之家论坛上的公开发帖。我们利用self-chat prompting方法引导ChatGPT围绕帖子主题生成多轮的问答。
在引导ChatGPT生成数据的过程中，我们通过精心设计的prompt确保生成的问答符合中国的国情、立场、态度和语言风格。

#### 金融任务指令
金融任务指令数据来源于两个部分：
（1）金融NLP数据集。该部分是基于已有的金融NLP数据集，通过人工编写的prompt改编而来的，图3就是一个例子。我们搜集了十余个开源的NLP中文数据集，可以分为情绪分析、信息抽取、文本生成、文本分类和翻译等几类。此数据集的分布如下所示：

| Dataset            | Major Task Type        | Minor Task Type           | \# Samples |
|--------------------|------------------------|---------------------------|-----------:|
| FPB                | Sentiment Analysis     | Sentiment Analysis        |      18690 |
| FIQA-SA            | Sentiment Analysis     | Sentiment Analysis        |          - |
| FNSC               | Sentiment Analysis     | Sentiment Analysis        |          - |
| CCKS-NEC-2022      | Imformation Extraction | Causality Extraction      |       7499 |
| SmoothNLP IEE      | Imformation Extraction | Event Extraction          |       3256 |
| SmoothNLP NHG      | Text Generation        | Text Generation           |       4642 |
| CCKS2022-event     | Text Classification    | Event Type Classification |       3578 |
| Minds14            | Text Classification    | Intent Prediction         |      59143 |
| Financial Report   | Imformation Extraction | Entity Extraction         |      61705 |
| OpenKG             | Imformation Extraction | Entity Extraction         |       7672 |
| OpenKG             | Imformation Extraction | Entity Extraction         |      67921 |
| FDDC2018           | Translation            | Terminology Translation   |        333 |
| Wealth-alpaca-lora | Question Answering     | Question Answering        |      41825 |

（2）金融无标签文本数据集。这是一个金融文本的阅读理解数据集。我们从东方财富网收集了共87k个文章，包括金融新闻和行业研报摘要。然后，基于这些无标签文本中的段落，我们利用ChatGPT得到指令对。

#### 金融计算指令
在金融计算中，表达式计算器、方程求解器、正态概率表、计数器四种工具可以帮助模型完成大多数的计算任务。四种工具各有不同的调用命令、输入和输出。例如，计算器的命令是**[Calculator(expression)→result]**。在这一部分，构建金融计算指令的目的就是训练模型在合适的时候调用这些工具解决数学问题。四个工具的定义如下表所示：

#### 检索增强指令
检索增强指令的构造分为三步。第一步，我们根据新闻和研报等金融文本构造金融分析问题。第二步，我们在知识库中检索与问题有关的文档，其中参考文档源于我们构建金融知识库，包含18k研报和69k金融新闻。第三步，我们将问题和参考资料结合在一起，生成问题的答案。在这个过程中，问题和答案是由ChatGPT通过Chain-of-Retrieval (CoR) prompting方法生成的。最终我们构建了一个由20k条检索增强指令组成的数据集，其中的指令涵盖了金融领域中主要的分析形式，包括行业分析、政策分析、投资建议、公司战略规划等。

我们开源了部分数据集，您可以访问这个[链接](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT)下载数据集。


<!-- 不同场景下的法律智能应用通常需要结合法律文本理解和生成的多种基本能力。为此，我们构建了一个高质量的监督微调数据集 DISC-Law-SFT，包括法律信息提取、判决预测、文档摘要和法律问题解答，确保覆盖不同司法应用场景。DISC-Law-SFT 包括两个子集，即 DISC-Law-SFT-Pair 和 DISC-Law-SFT-Triplet。前者旨在为 LLM 引入法律推理能力，后者则有助于提高模型利用外部知识的能力，具体的构建细节请参照我们的[技术报告](https://arxiv.org/abs/2309.11325)。数据集的分布如下所示：

<img src="" alt="" width=""/>

<table>
  <tr>
    <th>数据集</th>
    <th>对应任务/来源</th>
    <th>样本量</th>
    <th>对应情境</th>
  </tr>
  <tr>
    <td rowspan="10">DISC-Law-SFT-Pair</td>
    <td>司法要素提取</td>
    <td>32K</td>
    <td rowspan="7">法律专业人员助手</td>
  </tr>
  <tr>
    <td>司法事件检测</td>
    <td>27K</td>
  </tr>
  <tr>
    <td>案件分类</td>
    <td>20K</td>
  </tr>
  <tr>
    <td>判决预测</td>
    <td>11K</td>
  </tr>
  <tr>
    <td>类案匹配</td>
    <td>8K</td>
  </tr>
  <tr>
    <td>司法摘要</td>
    <td>9K</td>
  </tr>
  <tr>
    <td>舆情摘要</td>
    <td>6K</td>
  </tr>
  <tr>
    <td>法律问答</td>
    <td>93K</td>
    <td>法律咨询服务</td>
  </tr>
  <tr>
    <td>司法阅读理解</td>
    <td>38K</td>
    <td rowspan="2">法律考试助手</td>
  </tr>
  <tr>
    <td>法律考试</td>
    <td>12K</td>
  </tr>
  <tr>
    <td rowspan="2">DISC-Law-SFT-Triplet</td>
    <td>判决预测</td>
    <td>16K</td>
    <td>法律专业人员助手</td>
  </tr>
  <tr>
    <td>法律问答</td>
    <td>23K</td>
    <td>法律咨询服务</td>
  </tr>
  <tr>
    <td rowspan="2">General</td>
    <td>Alpaca-GPT4</td>
    <td>48K</td>
    <td rowspan="2">通用场景</td>
  </tr>
  <tr>
    <td>Firefly</td>
    <td>60K</td>
  </tr>
  <tr>
    <td>总计</td>
    <td colspan="3">403K</td>
  </tr>
</table> -->

<!-- 我们总共发布了近30万条训练数据，其中包括 DISC-Law-SFT-Pair 和DISC-Law-SFT-Triplet。您可以访问这个[链接](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT)下载数据集。

### 检索增强模块

我们在 DISC-LawLLM 的基础上增加了一个基于开源检索框架 [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) 的检索模块。我们的知识库目前包括法条库和法考题库。

* 法条库包含 800 多部国家地方法律、条例和规定，其中包括《宪法》、《刑法》、《行政诉讼法》、《保险法》、《劳动法》、《著作权法》、《民法典》、《专利法》、《专属经济区和大陆架法》、《中国人民解放军选举全国人民代表大会和县级以上地方各级人民代表大会代表的办法》、《反分裂国家法》、《出境入境边防检查条例》、《国务院关于鼓励台湾同胞投资的规定》、《境内外国人宗教活动管理规定》等。
* 法考题库包含 2.4 万道法律相关的考试题目。

在未来，我们会增加更加丰富的知识库。我们还将进一步深入探索检索增强的 DISC-LawLLM，包括但不限于检索器与 LLM 的联合训练机制，各位有兴趣可以与我们一起交流。 -->

## 推理和部署

开源版本的 DISC-FinLLM 是基于 [Baichuan-13B-Chat](https://github.com/baichuan-inc/Baichuan-13B) 进行LoRA微调训练得到的。您可以直接从 [Hugging Face](https://huggingface.co/ShengbinYue/DISC-LawLLM) 上下载我们的模型权重，或者根据下面的代码样例自动获取。推理前请安装依赖：

```
pip install -r requirements.txt
```

### Python

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

model_path = "ShengbinYue/DISC-FinLLM"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model.generation_config = GenerationConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, use_fast=False, trust_remote_code=True,
)

messages = [
    {"role": "user", "content": "生产销售假冒伪劣商品罪如何判刑？"},
]
response = model.chat(tokenizer, messages)
```

### 命令行工具

```
python cli_demo.py
```

### 网页 Demo

依靠 streamlit 工具运行以下命令，会在本地启动一个 web 服务，把控制台给出的地址输入浏览器即可访问：

```
streamlit run web_demo.py --server.port 8888
```

此外，目前版本的 DISC-FinLLM 是以 Baichuan-13B 作为基座的，您可以参照 [Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B) 的介绍来进行 int8 或 int4 量化推理部署以及 CPU 部署。

<!-- ## 模型微调

开发者可以对 DISC-FinLLM 进行微调使用。在此可以参照与 DISC-LawLLM 兼容的微调工具 [LLaMA Efficient Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) 或是我们的 [DISC-MedLLM](https://github.com/FudanDISC/DISC-MedLLM) 医疗大模型。我们以 [LLaMA Efficient Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) 为例给出**全量**和 **LoRA** 两种微调示例。

首先，下载 [LLaMA Efficient Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) 并按其要求[安装依赖](https://github.com/hiyouga/LLaMA-Efficient-Tuning#getting-started)。注意训练数据按照项目中的要求进行处理。下面我们给出两种微调场景下的脚本样例。

### 全量微调

我们在 8 * Nvidia A800 80 GB + deepspeed 的环境下进行了全量微调测试。训练启动脚本示例如下：

```
deepspeed --num_gpus=8 src/train_bash.py \
    --stage sft \
    --model_name_or_path S heng bin \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --template baichuan \
    --finetuning_type full \
    --output_dir path_to_your_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \ 
    --per_device_eval_batch_size 4 \ 
    --gradient_accumulation_steps 8 \ 
    --preprocessing_num_workers 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 2.0 \
    --dev_ratio 0.01 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --fp16 \
    --deepspeed deepspeed.json
```

`deep_speed.json` 配置示例如下：

```json
{
    "train_micro_batch_size_per_gpu": "auto",
    "zero_allow_untested_optimizer": true,
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "initial_scale_power": 16, 
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },  
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": false,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients" : true
    }
}
```

### LoRA 微调

我们在 4 * Nvidia A800 80G 显卡上进行了 LoRA 微调测试。训练启动脚本示例如下：

```
torchrun --nproc_per_node 4 src/train_bash.py \
    --stage sft \
    --model_name_or_path ShengbinYue/DISC-LawLLM \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --template baichuan \
    --finetuning_type lora \
    --lora_rank 8 \ 
    --lora_target W_pack \
    --output_dir path_to_your_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \ 
    --per_device_eval_batch_size 4 \ 
    --gradient_accumulation_steps 8 \ 
    --preprocessing_num_workers 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 1e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 2.0 \
    --dev_ratio 0.01 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --fp16
``` -->

## DISC-Fin-Eval-Benchmark

我们建立了一个全面的评估框架 —— DISC-Fin-Eval Benchmark，从各个角度严格评估我们的模型。该评估框架包括四个不同的组成部分，即：金融NLP任务、人类试题、资料分析和时事分析。这一评估框架全面地证明了我们模型能力和训练数据的有效性。您可以点击此[链接](./eval)使用我们的 DISC-Fin-Eval-Benchmark。

### 评测系统

#### 金融NLP任务评测
我们使用FinCUGE评估基准测试模型处理金融NLP任务的能力。这个评测一共包含八项任务，其中包括情感分析、关系抽取、文本摘要、文本分类、事件抽取和其他任务。我们通过提示模板将这个数据集改造为小样本（few-shot）形式，使用常用的准确度（accuracy）、F1和Rouge指标评价模型的表现，来衡量模型在金融领域中理解文本和生成相关回答的能力。**你可以在这里查看我们[金融NLP任务评测](https://github.com/FudanDISC/DISC-FinLLM/tree/main/eval/evaluator)**的具体内容。





<!-- ### 主观评测

在主观评测部分，我们采用问答题形式进行评估，模拟主观考试问题的过程。我们从法律咨询、在线论坛、与司法相关的出版物和法律文件中手工构建了一个高质量的测试集。我们用 GPT-3.5 Turbo 作为裁判模型来评估模型的输出，并基于标准答案用准确性、完整性和清晰度这三个标准提供 1-5 的评分。

主观题数据集从来源于法律咨询、网上发帖、司法相关出版物和法律文书中手动构建的一个高质量的测试集，其中包括 300 个示例，涵盖了法律知识问答、法律咨询和判决预测等场景。

**你可以在这里查看我们的[主观评测集](https://github.com/FudanDISC/DISC-LawLLM/tree/main/eval/data/subjective_eval)** -->

### 评测结果

客观题评测采用 few-shot 方式，结果（%）如下：

|        模型        |  NJE 单选   |  NJE 多选   |  PAE 单选   |  PAE 多选   |  CPA 单选   |  CPA 多选   | UNGEE 单选  | UNGEE 多选  |  PFE 单选   |  LBK 单选   |   平均   |
|:----------------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|     ChatGLM      |   31.66   |   1.08    |   27.97   |   2.90    |   37.06   |   13.33   |   39.69   |   20.69   |   37.65   |   42.91   |   24.66   |
|  Baichuan-Chat   |   31.47   |   10.15   |   29.66   |   8.70    |   35.53   |   19.17   |   50.00   |   27.59   |   53.12   |   53.45   |   30.78   |
| Chinese-Alpaca-2 |   25.70   |   10.15   |   30.51   |   11.59   |   32.99   |   19.17   |   40.94   |   21.84   |   44.12   |   43.27   |   26.73   |
|  GPT-3.5-turbo   |   36.50   |   10.58   |   37.29   |   17.03   | **42.13** | **21.67** | **51.25** | **28.74** |   53.53   |   54.18   |   34.10   |
|     LexiLaw      |   20.11   |   7.56    |   23.73   |   10.14   |   24.87   |   19.17   |   31.56   |   16.09   |   31.76   |   40.36   |   21.50   |
|      LawGPT      |   22.91   |   6.26    |   31.36   |   7.61    |   25.38   |   16.67   |   30.31   |   13.79   |   34.71   |   29.09   |   20.60   |
|   Lawyer LLaMa   |   35.75   |   5.62    |   32.20   |   6.52    |   29.95   |   13.33   |   32.50   |   14.94   |   39.41   |   39.64   |   25.05   |
|     ChatLaw      |   27.56   |   7.99    |   31.36   |   9.42    |   35.53   |   11.67   |   35.62   |   17.24   |   42.35   |   41.09   |   25.20   |
|   DISC-LawLLM    | **42.09** | **19.87** | **40.68** | **18.48** |   39.59   |   19.17   |   50.94   |   25.29   | **57.06** | **54.91** | **37.10** |

主观题评测分数为 1-5，结果如下：

|        模型        | 准确性  | 完整性  | 清晰性  |  平均  |
|:----------------:|:----:|:----:|:----:|:----:|
|     ChatGLM      | 2.64 | 2.75 | 3.23 | 2.87 |
|  Baichuan-Chat   | 3.22 | **3.34** | 3.18 | 3.25 |
| Chinese-Alpaca-2 | 3.13 | 3.23 | 3.17 | 3.17 |
|     LexiLaw      | 3.06 | 2.62 | 3.00 | 2.90 |
|      LawGPT      | 3.02 | 2.58 | 2.96 | 2.86 |
|   Lawyer LLaMa   | 3.13 | 2.83 | 3.35 | 3.10 |
|     ChatLaw      | 3.31 | 2.90 | 3.35 | 3.19 |
|   DISC-LawLLM    | **3.46** | 3.12 | **3.59** | **3.39** |

## 致谢

本项目基于如下开源项目展开，在此对相关项目和开发人员表示诚挚的感谢：

- [**Baichuan-13B**](https://github.com/baichuan-inc/Baichuan-13B)
- [**Langchain-Chatchat**](https://github.com/chatchat-space/Langchain-Chatchat)
- [**LLaMA Efficient Tuning**](https://github.com/hiyouga/LLaMA-Efficient-Tuning)
- [**FireFly**](https://github.com/yangjianxin1/Firefly)

同样感谢其他限于篇幅未能列举的为本项目提供了重要帮助的工作。

## 声明

DISC-LawLLM 有着目前大语言模型尚无法克服的问题和缺陷，尽管它能够在许多任务和情境上提供法律服务，但模型应当仅供用户参考使用，并不能替代专业律师和法律专家，我们希望 DISC-LawLLM 的用户以批判性的眼光去评估模型。我们不对因使用 DISC-LawLLM 所引发的任何问题、风险或不良后果承担责任。

## 引用

如果我们的项目对您的研究和工作有帮助，请如下引用我们的项目：

```
@misc{yue2023disclawllm,
    title={DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services}, 
    author={Shengbin Yue and Wei Chen and Siyuan Wang and Bingxuan Li and Chenchen Shen and Shujun Liu and Yuxuan Zhou and Yao Xiao and Song Yun and Xuanjing Huang and Zhongyu Wei},
    year={2023},
    eprint={2309.11325},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## 协议

DISC-LawLLM 可在 Apache 许可证下使用。请查看 [LICENSE](./LICENSE) 文件获取更多信息。


## Star History

<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=FudanDISC/DISC-LawLLM&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=FudanDISC/DISC-LawLLM&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=FudanDISC/DISC-LawLLM&type=Date" />
</picture>
