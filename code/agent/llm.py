import os
from openai import OpenAI
import argparse
import time
import json
from tqdm import tqdm
from typing import List, Dict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

from evaluator import Evaluator
from utils.config_utils import Config
from utils.json_utils import load_json, preprocess_response_string
from utils.pdg_utils import ReadUtils, ProcessUtils

class LLMType(Enum):
    DEEPSEEK = {
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat"
    }

LLM_MODEL = {}
class PromptTemplate(Enum):
    EA_IA = {
        'system_prompt': """You are an expert in comprehensive code dependency analysis, specializing in both explicit dependencies (e.g., data and control flow) and implicit dependencies (e.g., contextual relevance, naming similarity, structural similarity, code clones, and refactorings). Given: - A Program Dependency Graph (PDG) where each node represents a changed code segment (based on a commit diff), and edges (compressed) indicate connectivity (meaning two nodes are connected if there was a dependency path between them in the original PDG). - A context object containing information from the Program Dependence Graph (PDG), including changed nodes and their relationship and neighboring nodes. - A commit diff in the following format: {{filename: [(symbol (- or +), linenumber, linecontent), ...]}}.  Task: - Group code changes such that each group contains changes connected by explicit or implicit dependencies 
- Ensure that changes in different groups are independent from each other. - Grouping must be done based on purpose, not file paths alone. For instance, changes in the same file may still belong to different concerns.  The following principles are provided to guide your analysis: - Code changes with data dependencies often belong to the same concern. - Code changes with control dependencies often belong to the same concern. - Code changes with semantic similarity may belong to the same concern. - Code changes with high textual or structure similarity may belong to the same concern. - Code changes introduced for cosmetic edits, such as syntactic formatting, refactoring, or non-functional textual modifications, often belong to the same concern.
Output format (JSON): ```json
{{
  "answer": {{filename: [(symbol (- or +), linenumber, groupnumber), ...]}},
  "explanation": "Explain your grouping decisions, highlighting key dependencies."
}}  
""",
        'user_prompt': """ACTUAL_INPUT_HERE GRAPHS_SENTENCE_HERE FILE_CONTENT_SENTENCE_HERE Please analyze the commit diff using the PDG and context, output the grouping in JSON format with 'answer' and 'explanation' fields. Remember: group changes based on both explicit (data/control) and implicit (contextual/purpose/refactoring) relationships, not file paths."""
    } 
    EA_IA_Cot = {
        'system_prompt': """You are an expert in comprehensive code dependency analysis, specializing in both explicit dependencies (e.g., data and control flow) and implicit dependencies (e.g., contextual relevance, naming similarity, structural similarity, code clones, and refactorings). Given: - A Program Dependency Graph (PDG) where each node represents a changed code segment (based on a commit diff), and edges (compressed) indicate connectivity (meaning two nodes are connected if there was a dependency path between them in the original PDG). - A context object containing information from the Program Dependence Graph (PDG), including changed nodes and their relationship and neighboring nodes. - A commit diff in the following format: {{filename: [(symbol (- or +), linenumber, linecontent), ...]}}.  Task: - Group code changes such that each group contains changes connected by explicit or implicit dependencies
- Ensure that changes in different groups are independent from each other. - Grouping must be done based on purpose, not file paths alone. For instance, changes in the same file may still belong to different concerns. The following principles are provided to guide your analysis: - Code changes with data dependencies often belong to the same concern. - Code changes with control dependencies often belong to the same concern. - Code changes with semantic similarity may belong to the same concern. - Code changes with high textual or structure similarity may belong to the same concern. - Code changes introduced for cosmetic edits, such as syntactic formatting, refactoring, or non-functional textual modifications, often belong to the same concern.
Output format (JSON): ```json
{{
  "answer": {{filename: [(symbol (- or +), linenumber, groupnumber), ...]}},
  "explanation": "Explain your grouping decisions, highlighting key dependencies."
}}  
""",
        'user_prompt': """ACTUAL_INPUT_HERE GRAPHS_SENTENCE_HERE FILE_CONTENT_SENTENCE_HERE Please analyze the commit diff using the PDG and context, output the grouping in JSON format with 'answer' and 'explanation' fields. Remember: group changes based on both explicit (data/control) and implicit (contextual/purpose/refactoring) relationships, not file paths. Let's think step by step."""
    }
    EA = {
        'system_prompt': f"""
You are an expert in code dependency analysis, specializing in analyzing explicit dependencies between code changes — including both data dependencies (such as variable or value flows) and control dependencies (such as conditional structures and execution order). 
Given: - A Program Dependency Graph (PDG) where each node represents a changed code segment (based on a commit diff), and edges (compressed) indicate connectivity (meaning two nodes are connected if there was a dependency path between them in the original PDG). - A commit diff in the following format: {{filename: [(symbol (- or +), linenumber, linecontent), ...]}}.  Task: - Group related code changes into groups, where changes in the same group are highly related via data or control dependencies.
- Ensure that changes in different groups are independent from each other. - Ignore file boundaries when grouping even if changes are in the same file, they can belong to different groups if they are not data/control-dependent. The following principles are provided to guide your analysis: - Code changes with data dependencies often belong to the same concern. - Code changes with control dependencies often belong to the same concern.
Output format (JSON): ```json
{{
  "answer": {{filename: [(symbol (- or +), linenumber, groupnumber), ...]}},
  "explanation": "Explain your grouping decisions, highlighting key dependencies."
}}""",
        'user_prompt': "ACTUAL_INPUT_HERE GRAPHS_SENTENCE_HERE Please analyze the commit diff using the PDG and output the grouping in JSON format with 'answer' and 'explanation' fields. Remember: group changes based on data and control dependencies, not file paths."
    }
    IA = {
        'system_prompt': """
You are an expert in code dependency analysis. Your task is to analyze implicit dependencies between code changes, including contextual relationships, naming and structural similarities, code clones, and refactorings that might implicitly connect changes even if explicit data or control dependencies are not evident.
Given: - A context object containing information from the Program Dependence Graph (PDG), including changed nodes and their relationship and neighboring nodes. - A commit diff in the following format: {{filename: [(symbol (- or +), linenumber, linecontent), ...]}}.  Task: - Group related code changes into groups, where changes in the same group are highly related. - Ensure that changes in different groups are independent from each other. - Do not group changes solely by file paths; prioritize logical purpose, context, and relationships. The following principles are provided to guide your analysis: - Code changes with semantic similarity may belong to the same concern. - Code changes with high textual or structure similarity may belong to the same concern. - Code changes introduced for cosmetic edits, such as syntactic formatting, refactoring, or non-functional textual modifications, often belong to the same concern.
Output format (JSON): ```json
{{
  "answer": {{filename: [(symbol (- or +), linenumber, groupnumber), ...]}},
  "explanation": "Explain your grouping decisions, highlighting key dependencies."
}}""",
        'user_prompt': """ACTUAL_INPUT_HERE FILE_CONTENT_SENTENCE_HERE Please analyze the commit diff using and output the grouping in JSON format, including 'explanation' and 'answer' fields. Group changes based on their purpose and implicit relationships (naming, structure, refactoring), rather than just file paths."""
    }


class Commit_Decomposer:
    def __init__(self,
                 repo_filter: List[str] = None,
                 config: Config = None,             
                 max_samples: int = None,
                 set_concern_num: bool = False,
                 user_prompt: str = None,
                 system_prompt: str = None,
                 add_graphs: bool = False,
                 add_file_content: bool = False,
                 repeat_dict: dict = None,):
        """
        Initialize the Commit Decomposer
        
        :param repo_filter: List of repository names to process (None processes all)
        :param model_call: Function for calling LLM API (prompt, max_tokens) -> response
        :param base_dir: Base directory for data storage
        :param max_samples: Maximum number of samples to process per repository
        :param user_prompt: Default prompt template for decomposition requests
        """
        self.repo_filter = set(repo_filter) if repo_filter else None
        self.config = config or Config()
        self.base_dir = config.base_data_dir or ''
        self.result_dir = config.base_result_dir or ''
        self.clean_result_dir = config.base_clean_result_dir or ''
        self.max_samples = max_samples
        self.processed_data = []
        self.set_concern_num = set_concern_num
        self.user_prompt = user_prompt or ''
        self.system_prompt = system_prompt or ''
        self.add_graphs = add_graphs
        self.add_file_content = add_file_content
        self.repeat_dict = repeat_dict or {}

    def load_data(self) -> List:
        """Load data from configured directories"""
        data_list = []
        all_num = 0
        for repo_name in os.listdir(self.base_dir):
            if self.repo_filter and repo_name not in self.repo_filter:
                continue
            repo_path = os.path.join(self.base_dir, repo_name)
            if not os.path.isdir(repo_path):
                continue
            chunk_dirs = os.listdir(repo_path)
            for chunk_dir in tqdm(chunk_dirs, desc=f"Loading data of {repo_name}", unit="chunk"):
                if self.max_samples and all_num >= self.max_samples:
                    break
                cor_result_path = os.path.join(self.result_dir, f"{repo_name}_{chunk_dir}.json")
                if os.path.exists(cor_result_path):
                    print(f"Skipping {cor_result_path} as it exists")
                    continue

                diff_file_data = {
                    "repo": repo_name,
                    "chunk_dir": chunk_dir
                }
                
                data_list.append(diff_file_data)
                all_num += 1
        print("Loaded data from all repositories")
        return data_list

class BaseAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.client = OpenAI(
            api_key=LLM_MODEL["api_key"],
            base_url=LLM_MODEL["base_url"]
        )
        self.model_name = LLM_MODEL["model_name"]
        self.memory = []

    def call_llm(self, messages) -> str:
        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"} ,
                    extra_body={"enable_thinking": False},
                    stream=True,
                )
                response_chunks = []
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        response_chunks.append(chunk.choices[0].delta.content)

                response = "".join(response_chunks)
                return response
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"LLM API call failed after {max_retries} attempts: {e}")
                time.sleep(0.5)  # Brief pause before retrying

def generate_prompt_template(llm_type: str):
    if llm_type == "EA":
        user_template = PromptTemplate.EA.value['user_prompt']
        system_template = PromptTemplate.EA.value['system_prompt']
    elif llm_type == "IA":
        user_template = PromptTemplate.IA.value['user_prompt']
        system_template = PromptTemplate.IA.value['system_prompt']
    elif llm_type == "EA_IA":
        user_template = PromptTemplate.EA_IA.value['user_prompt']
        system_template = PromptTemplate.EA_IA.value['system_prompt']
    elif llm_type == "EA_IA_Cot":
        user_template = PromptTemplate.EA_IA_Cot.value['user_prompt']
        system_template = PromptTemplate.EA_IA_Cot.value['system_prompt']
    return [system_template, user_template]

def check_args(llm_type, add_graphs, add_file_content, prompt_category) -> str:
    if llm_type == 'EA' and (not add_graphs or add_file_content):
        return "Error: EA requires add_graphs to be True and add_file_content to be False"
    if llm_type == 'IA' and (add_graphs or not add_file_content):
        return "Error: IA requires add_graphs to be False and add_file_content to be True"
    if llm_type in ['EA_IA', 'EA_IA_Cot'] and (not add_graphs or not add_file_content):
        return "Error: EA_IA requires add_graphs and add_file_content to be True"
    if llm_type == 'EA_IA_Cot' and prompt_category != 1:
        return "Error: EA_IA_Cot requires prompt_category to be 1"
    return 'success'

def generate_prompt(user_template: str = None, system_template: str = None, item: dict = {}, add_graphs: bool = False, add_file_content: bool = False) -> List[Dict[str, str]]:
    user_prompt =  user_template 
    system_prompt = system_template 

    diff_sentence = f"Commit diffs: {str(item['merged_diff'])}"
    user_prompt = user_prompt.replace("ACTUAL_INPUT_HERE", diff_sentence)

    graphs_sentence = f"PDGs: {item['pdg_contents']}"
    user_prompt = user_prompt.replace("GRAPHS_SENTENCE_HERE", graphs_sentence)

    file_content_sentence = f"Contexts: {item['file_contents']}"
    user_prompt = user_prompt.replace("FILE_CONTENT_SENTENCE_HERE", file_content_sentence)


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return messages

def parse_structured_output(response_text: str) -> Dict[str, str]:
    """
    Parse LLM response to extract structured output.

    Args:
        response_text: Text response from LLM

    Returns:
        Dictionary containing structured fields
    """
    try:
        # Try parsing as JSON
        parsed = json.loads(preprocess_response_string(response_text))
        return parsed
    except json.JSONDecodeError:
        # If not valid JSON, extract from text
        # This is a fallback for when the model doesn't format JSON correctly
        lines = response_text.strip().split('\n')
        result = {}

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                result[key] = value

        # Ensure explanation and answer fields exist
        if "answer" not in result:
            result["answer"] = "No structured answer found in response"

        return result

def process_item(item, ct: Commit_Decomposer, model_name: str):
    repo_name = item["repo"]
    chunk_dir = item["chunk_dir"]

    merged_contents, pdg_contents, context_contents = [], None, None
    try:
        chunk_path = os.path.join(base_data_dir, repo_name, chunk_dir)
        merged_diffs_path = os.path.join(chunk_path, 'tangled_commit')
        contexts_path_dir = os.path.join(chunk_path, 'implicit_contexts')
        pdgs_path_dir = os.path.join(chunk_path, 'explicit_contexts')

        merged_file_name = os.listdir(merged_diffs_path)[0]
        merged_contents = json.dumps(load_json(os.path.join(merged_diffs_path, merged_file_name)), ensure_ascii=False)

        pdg_file_path = os.path.join(pdgs_path_dir, 'explicit_contexts.dot')
        if os.path.exists(pdg_file_path):
            pdg_contents = ReadUtils.obj_dict_to_networkx(ReadUtils.read_graph_from_dot(pdg_file_path))
            pdg_contents = ProcessUtils.nx_to_str(pdg_contents) if pdg_contents else None,
        
        context_file_path = os.path.join(contexts_path_dir, 'implicit_contexts.dot')

        if os.path.exists(context_file_path):
            context_contents = ReadUtils.obj_dict_to_networkx(ReadUtils.read_graph_from_dot(context_file_path))
            context_contents = ProcessUtils.nx_to_str(context_contents) if context_contents else None

    except Exception as e:
        print(f"Error loading data for {repo_name}_{chunk_dir}: {e}")
        return
    
    chunk_item = {
        'repo': repo_name,
        'chunk_dir': chunk_dir,
        'merged_diff': merged_contents,
        'pdg_contents': pdg_contents,
        'file_contents': context_contents,
    }

    messages = generate_prompt(
        user_template=ct.user_prompt, system_template=ct.system_prompt, item=chunk_item, add_graphs=ct.add_graphs, add_file_content=ct.add_file_content
    )
    os.makedirs(ct.result_dir, exist_ok=True)
    result_file = os.path.join(ct.result_dir, f"{repo_name}_{chunk_dir}.json")

    # JUDGE REPEAT - if the result exists, skip it
    if os.path.exists(result_file):
        return

    model_agent =  BaseAgent(chunk_dir)
    
    llm_response = model_agent.call_llm(
        chunk_dir=chunk_dir,
        repo_name=repo_name,
        messages=messages,
        model_name=model_name
    )
    # Parse response
    try:
        result = json.loads(preprocess_response_string(llm_response))
        print(f'{repo_name}_{chunk_dir} reponse successfully parsed')
    except json.JSONDecodeError:
        # If JSON format is not correct, use fallback parsing
        print(f"{repo_name}_{chunk_dir} , using fallback parsing")
        result = parse_structured_output(llm_response)

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)



def process_repository(repo_data: list,
                        thread_num: int, ct: Commit_Decomposer, model_name: str) -> list:
    with tqdm(total=len(repo_data), desc="Processing Repos", unit="case") as pbar:
        with ThreadPoolExecutor(max_workers=thread_num) as executor:
            futures = [executor.submit(process_item, item, ct, model_name) for item in repo_data]
            for future in as_completed(futures):
                pbar.update(1)


def untangle(ct: Commit_Decomposer, model_name: str):
    data = ct.load_data()
    process_repository(data, 10, ct, model_name)

def evalute(evaluator: Evaluator):
    evaluator.call_mal()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Untangle commits with different input configurations.")

    parser.add_argument("--add_comments", type=int, choices=[0,1],
                        help="Whether to add comments (0 or 1).")
    parser.add_argument("--add_explicit_context", type=int, choices=[0,1],
                        help="Whether to add explicit context (0 or 1).")
    parser.add_argument("--add_implicit_context", type=int, choices=[0,1],
                        help="Whether to add implicit context (0 or 1).")
    parser.add_argument("--model_name", choices=["deepseek-v3"],
                        help='Model name to use. Only "deepseek-v3" is supported.')
    parser.add_argument("--llm_type", choices=["EA", "IA", "EA_IA", "EA_IA_Cot"],
                        help="Type of LLM. Choose from 'EA', 'IA', 'EA_IA', 'EA_IA_Cot'.")

    args = parser.parse_args()

    add_comments = bool(args.add_comments)
    add_graphs = bool(args.add_explicit_context)
    add_file_content = bool(args.add_implicit_context)
    model_name = args.model_name
    llm_type = args.llm_type
    prompt_category = 1 if llm_type == "EA_IA_Cot" else 0

    
    LLM_MODEL = LLMType.DEEPSEEK.value

    config = Config(add_comments=add_comments, add_graphs=add_graphs, add_file_content=add_file_content, add_concern_num = prompt_category, ma_model_category='deepseek-v3')
    base_data_dir = config.base_data_dir
    base_result_dir = config.base_result_dir
    # FILTER UNTAGGLE REPOS
    repo_filter = config.llm_repo_filter

    if not os.path.exists(base_result_dir):
        os.makedirs(base_result_dir)


    system_prompt, user_prompt = generate_prompt_template(llm_type)
    decomposer = Commit_Decomposer(
        repo_filter=repo_filter,
        set_concern_num=False,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        add_graphs=add_graphs, 
        add_file_content=add_file_content,
        config=config,
    )
    
    evaluator = Evaluator(base_dir=base_data_dir, result_dir=base_result_dir, evaluation_file_path=re.sub(r'(.*)/[^/]*/[^/]*$', r'\1', base_result_dir) + '/evaluation_result.json')
   
    # UNTANGLE COMMITS
    untangle(ct=decomposer, model_name=model_name)
    # EVALUATE RESULTS
    evalute(evaluator)   
