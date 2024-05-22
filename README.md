# LLM as Interpreter for Natural Language Programming, Pseudo-code Programming and Flow Programming of AI Agents

Since their inception, programming languages have trended towards greater readability and lower barriers for programmers. Following this trend, natural language can be a promising type of programming language that provides great flexibility and usability and helps towards the democracy of programming. However, the inherent vagueness, ambiguity, and verbosity of natural language pose significant challenges in developing an interpreter that can accurately understand the programming logic and execute instructions written in natural language. Fortunately, recent advancements in Large Language Models (LLMs) have demonstrated remarkable proficiency in interpreting complex natural language. Inspired by this, we develop a novel system for Code Representation and Execution (CoRE), which employs LLM as interpreter to interpret and execute natural language instructions. The proposed system unifies natural language programming, pseudo-code programming, and flow programming under the same representation for constructing language agents, while LLM serves as the interpreter to interpret and execute the agent programs. In this paper, we begin with defining the programming syntax that structures natural language instructions logically. During the execution, we incorporate external memory to minimize redundancy. Furthermore, we equip the designed interpreter with the capability to invoke external tools, compensating for the limitations of LLM in specialized domains or when accessing real-time information.

This package is mainly contributed by [Shuyuan Xu](https://github.com/shuyuan-x) (shuyuan.xu@rutgers.edu), [Zelong Li](https://github.com/lzl65825) (zelong.li@rutgers.edu), and [Yongfeng Zhang](https://github.com/evison) (yongfeng.zhang@rutgers.edu). We welcome any issues and requests for model implementation and bug fix.

## Reference

To be updated

## Requirements

- Python==3.9
- PyTorch==2.2.0
- transformers==4.40.2
- langchain==0.1.4
- peft==0.7.1

## Preparation

0. Clone this repo.

1. Create a conda virtual environment and install the Pytorch matching your CUDA version. For example, for CUDA version 12.1:

```
conda create -n your_env_name python=3.9
conda activate your_env_name

conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

2. Install necessary packages:

```
pip install -r requirements.txt
```

3. Download the OpenAGI data from this [Google Drive link](https://drive.google.com/drive/folders/1AjT6y7qLIMxcmHhUBG5IE1_5SnCPR57e?usp=share_link), put it into the *FlowProgramming/* folder, then unzip it.

4. Download the [database](https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link) and unzip it to the `CoRE` directory (i.e., `your/path/CoRE`) and rename it as `travel_database`.

5. Make sure you are in the *FlowProgramming/src* folder before running the codes. Otherwise,

```
cd src
```

## Running Command Examples

OpenAGI on gpt-4-1106-preview:
```commandline
python main.py \
--flow_name=OpenAGI_Flow.txt \
--tool_name=tools.txt \
--task=OpenAGI \
--log_file_name=OpenAGI_gpt_log.txt \
--model_name=gpt-4-1106-preview \
--openai_key="YOUR OPENAI KEY"
```

OpenAGI on TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ:
```commandline
python main.py \
--flow_name=OpenAGI_Flow.txt \
--tool_name=tools.txt \
--task=OpenAGI \
--log_file_name=OpenAGI_mixtral_log.txt \
--model_name=TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ \
--openai_key="YOUR OPENAI KEY"
```

TravelPlanner on gpt-4-1106-preview:
```commandline
python main.py \
--flow_name=TravelPlanner_Flow.txt \
--tool_name=tools.txt \
--task=TravelPlanner \
--log_file_name=TravelPlanner_gpt_log.txt \
--results_name=gpt \
--model_name=gpt-4-1106-preview \
--openai_key="YOUR OPENAI KEY"
```

TravelPlanner on TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ:
```commandline
python main.py \
--flow_name=TravelPlanner_Flow.txt \
--tool_name=tools.txt \
--task=TravelPlanner \
--log_file_name=TravelPlanner_mixtral_log.txt \
--results_name=mixtral \
--model_name=TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ \
--openai_key="YOUR OPENAI KEY"
```

## Reference

- We leveraged the dataset of [OpenAGI](https://github.com/agiresearch/OpenAGI) projects to implement our experiment.
