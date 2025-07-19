# Supplemental Materials

This repository contains the replication package for the paper *ColaUntangle: LLM-Driven Collaborative Model for Untangling Commits via Explicit and Implicit Dependency Reasoning*.

## Introduction

We have organized the replication package into four main folders to provide a clear and systematic structure for reproducing our study.

The file directory tree is as follows:

```
├─ 📁agent
│  ├─ 📄evaluator.py
│  ├─ 📄llm.py
│  ├─ 📄multi_agent.py
│  └─ 📄multi_agent_wo_tools.py
├─ 📁preprocess_corpus
│  ├─ 📄create_explicit_contexts.py
│  └─ 📄create_implicit_contexts.py
├─ 📁statistics
│  └─ 📄run.py
└─ 📁utils
   ├─ 📄config_utils.py
   ├─ 📄json_utils.py
   └─ 📄pdg_utils.py
data
└─ 📁corpora_raw
   └─ 📄stmt_num.json
tools
├─ 📁JavaExtractors
└─ 📁CsharpExtractors
results
├─ 📁llm
├─ 📁multi_agent
├─ 📁figs
└─ 📁tables
```

### **code**

The folder `code` contains the implementation of *ColaUntangle*, which consists of four components: `agent`, `preprocess_corpus`, `statistics`, and `utils`. Their functions are as follows:

- **agent**: Contains the implementation of the collaborative consultation process in *ColaUntangle*. Specifically, `multi_agent.py` and `multi_agent_wo_tools.py` are used to run experiments for *ColaUntangle* and its variants initialized with different multi-agent strategies.
- **preprocess_corpus**: Includes scripts to construct structured code information required by *ColaUntangle*. This folder contains files for extracting explicit and implicit contexts from tangled commits.
- **statistics**: The script `run.py` is used to calculate and display all statistical results from our experiments.
- **utils**: Contains configuration files used by *ColaUntangle*.

### **data**

The folder `data` is used to store the dataset utilized in our study. Due to the large size of the full dataset, only a statistical summary file is provided in this package. The complete dataset can be downloaded from the following link:  [ColaUntangle Dataset](https://figshare.com/s/ae92eae6ab5b52182075)

### **tools**

The folder `tools` contains the extractors used for extracting explicit and implicit contexts.

### **results**

The folder `results` contains the experimental outputs. Specifically:

- `figs` and `tables` store the figures and tables summarizing our results.
- `llm` and `multi_agent` contain the untangling results produced by *ColaUntangle* and its variants on the dataset. Each subfolder within `llm` and `multi_agent` corresponds to a specific variant of *ColaUntangle*.

Due to the large size of the untangling results, only the statistical summary files are included in each variant subfolder. The complete results can be requested via the following link:

[ColaUntangle Results](https://figshare.com/s/319067b8b3a8575a2186)

## Usage

To reproduce our work, please follow these four steps: Dataset Download, Environment Setup, Execute *ColaUntangle*, and Statistical Analysis.

### Dataset Download

The dataset can be acquired from the following link:   [ColaUntangle Dataset](https://figshare.com/s/ae92eae6ab5b52182075)

After downloading the dataset, replace the `data` folder in this package with your downloaded dataset folder.

The provided dataset includes the preprocessed explicit and implicit contexts for each tangled commit. If you wish to reproduce the process of extracting explicit and implicit contexts, please refer to the *Execute ColaUntangle* section.

### Environment Setup

The environment setup for *ColaUntangle* consists of two parts: execution environment and extractor environment.

#### **Execution Environment**

To successfully execute our Python programs, please install the following dependencies:

```
numpy==2.0.2
pandas==2.2.3
tqdm==4.67.1
matplotlib==3.9.2
scipy==1.15.2
openai==1.97.0
networkx==3.4.2
rapidfuzz==3.13.0
pydot==3.0.4
pygraphviz==12.2.1
```

All dependencies can be installed via the `pip` package manager, except for `pygraphviz`. To install `pygraphviz`, download the setup file from [Graphviz GitLab Release](https://gitlab.com/graphviz/graphviz/-/package_files/164443457/download) and run the following command (adjust the include and lib paths to match your local environment):

```bash
python -m pip install --config-settings="--global-option=build_ext" --config-settings="--global-option=-IC:\environment\pygraphviz\Graphviz\include" --config-settings="--global-option=-LC:\environment\pygraphviz\Graphviz\lib" pygraphviz
```

#### **Extractor Environment**

If you wish to reproduce the extraction of explicit and implicit contexts, you will need to run the Python scripts in the `preprocess_corpus` folder, which internally call the extractors located in the `tools` folder. Therefore, additional configurations are required. If you directly utilize the processed contexts provided in the downloaded dataset, you can skip this step.

- To successfully call the C# extractor, a Windows system with .NET Framework 4.5 installed is required. You can download .NET Framework 4.5 from the [Official Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=30653).
- To successfully call the Java extractor, you need to install a Java Runtime Environment (JRE, version 8 or newer). The latest JRE version for your platform can be downloaded from [java.com](https://java.com/).

### Execute *ColaUntangle*

To execute *ColaUntangle*, you need to follow two main steps: preprocessing the corpus (optional) and running *ColaUntangle*.

#### **Preprocess the Corpus**

If you wish to reproduce our process of generating explicit and implicit contexts, please set up the extractor environment as described in the previous section. Additionally, you need to clone the repositories used in our dataset, as they are required to extract the program dependency graphs, which are prerequisites for generating explicit and implicit contexts.

To prepare the repositories, clone the following projects:

| Project                                                      | Last Revision |
| ------------------------------------------------------------ | ------------- |
| [Commandline](https://github.com/commandlineparser/commandline) | 67f77e1       |
| [CommonMark](https://github.com/Knagis/CommonMark.NET)       | f3d5453       |
| [Hangfire](https://github.com/HangfireIO/Hangfire)           | 175207c       |
| [Humanizer](https://github.com/Humanizr/Humanizer)           | 604ebcc       |
| [Lean](https://github.com/QuantConnect/Lean)                 | 71bc0fa       |
| [Nancy](https://github.com/NancyFx/Nancy)                    | dbdbe94       |
| [Newtonsoft.Json](https://github.com/JamesNK/Newtonsoft.Json) | 4f8832a       |
| [Ninject](https://github.com/ninject/ninject)                | 6a7ed2b       |
| [RestSharp](https://github.com/restsharp/RestSharp)          | b52b9be       |
| [Druid](https://github.com/apache/druid) | 26b568        |
| [Dubbo]()                                                    | c39389        |
| [Elasticsearch](https://github.com/elastic/elasticsearch) | f1f745        |
| [EventBus](https://github.com/greenrobot/EventBus) | 019492        |
| [Ghidra](https://github.com/NationalSecurityAgency/ghidra) | 5e825e        |
| [Guava](https://github.com/google/guava) | 41c7d2        |
| [Retrofit](https://github.com/square/retrofit)| c4cfb1        |
| [RxJava](https://github.com/ReactiveX/RxJava)                | 93aa12        |
| [Spring-boot](https://github.com/spring-projects/spring-boot) | 86a8c9        |
| [Zxing](https://github.com/zxing/zxing) | 2a779a        |

For each project, execute:

```bash
git clone <repository_url> ./subjects/<project_name>
cd ./subjects/<project_name>
git reset --hard <Last Revision>
```

After preparing the repositories, run `create_explicit_contexts.py` in the `preprocess_corpus` folder, note that you should generate explicit contextes before implicit contexts as implicit contexts rely on the PDGs generated by  `create_explicit_contexts.py`  :

```bash
[../ColaUntangle/code] $ python preprocess_corpus/create_explicit_contexts.py comment language
```

- `comment` (0/1): indicates whether to process the dataset with (1) or without (0) comments.
- `language` (csharp/java): specifies the dataset language to process.

Then, run `create_implicit_contexts.py` in the same folder:

```bash
[../ColaUntangle/code] $ python preprocess_corpus/create_implicit_contexts.py comment language
```

#### Run ColaUntangle

To execute *ColaUntangle*, run the scripts located in the `agent` folder.

First, fill in the secret key of the Large Language Model (LLM) API you intend to use in `multi_agent.py`. For example:

```python
class LLMType(Enum):
    DEEPSEEK = {
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat"
    }
```

Then, use the following command to run `multi_agent.py`:

```bash
[../ColaUntangle/code] $ python ./agent/multi_agent.py --add_comments 1 --agent_model_config deepseek-v3
```

- `add_comments` [0, 1]: specifies whether to run *ColaUntangle* on the dataset with (1) or without (0) comments.
- `agent_model_config` ["deepseek-v3", "chatgpt-4o", "claude4-sonet", "qwen", "o4-mini"]: specifies which LLM model to use for initializing *ColaUntangle*.

We provide various parameters to execute different variants of *ColaUntangle*. Below are some examples:

##### **ColaUntangle_w/o_comment**

Runs *ColaUntangle* on the dataset without comments.

```bash
[../ColaUntangle/code] $ python ./agent/multi_agent.py --add_comments 0 --agent_model_config deepseek-v3
```

##### **ColaUntangle_Chatgpt_4o**

Initializes *ColaUntangle* with ChatGPT-4o.

```bash
[../ColaUntangle/code] $ python ./agent/multi_agent.py --add_comments 1 --agent_model_config chatgpt-4o
```

##### **ColaUntangle_Claude-4-sonet**

Initializes *ColaUntangle* with Claude-4-sonet.

```bash
[../ColaUntangle/code] $ python ./agent/multi_agent.py --add_comments 1 --agent_model_config claude4-sonet
```

To run the *ColaUntangle_w/o_info_tools* variant, use:

```bash
[../ColaUntangle/code] $ python ./agent/multi_agent_wo_tools.py --add_comments 1 --agent_model_config deepseek-v3
```

#### Running Variants on a Single LLM

To execute the LLM_zeroshot variant, run:

```bash
[../ColaUntangle/code] $ python ./agent/llm.py --add_comments 1 --add_explicit_context 1 --add_implicit_context 1 --model_name deepseek-v3 --llm_type EA_IA
```

- `add_comments` [0, 1]: specifies whether to run *ColaUntangle* on the dataset with (1) or without (0) comments.
- `add_explicit_context` [0, 1]: specifies whether to provide explicit contexts.
- `add_implicit_context` [0, 1]: specifies whether to provide implicit contexts.
- `model_name` ["deepseek-v3"]: specifies which model to initialize *ColaUntangle*.
- `llm_type` ["EA", "IA", "EA_IA", "EA_IA_Cot"]: specifies the variant to run.

We demonstrate the variants and their corresponding commands below:

##### **LLM_zeroshot_CoT**

```bash
[../ColaUntangle/code] $ python ./agent/llm.py --add_comments 1 --add_explicit_context 1 --add_implicit_context 1 --model_name deepseek-v3 --llm_type EA_IA_Cot
```

##### **ColaUntangle w/o Collaborative w/o Explicit Worker Agent**

```bash
[../ColaUntangle/code] $ python ./agent/llm.py --add_comments 1 --add_explicit_context 0 --add_implicit_context 1 --model_name deepseek-v3 --llm_type IA
```

##### **ColaUntangle w/o Collaborative w/o Implicit Worker Agent**

```bash
[../ColaUntangle/code] $ python ./agent/llm.py --add_comments 1 --add_explicit_context 1 --add_implicit_context 0 --model_name deepseek-v3 --llm_type EA
```

### Evaluate

To evaluate the performance of *ColaUntangle*, run the following script in the `statistics` folder:

```bash
[../ColaUntangle/code] $ python ./statistics/run.py
```

The performance results and all statistical data will be saved in the `results` folder.