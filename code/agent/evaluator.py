import os
import json
from typing import Dict
import numpy as np
import tqdm
from scipy.optimize import linear_sum_assignment
from utils.json_utils import load_json
from utils.config_utils import Config

class Evaluator:
    def __init__(self, base_dir: str, result_dir: str, evaluation_file_path):
        self.base_dir = base_dir
        self.result_dir = result_dir
        self.evaluation_file_path = evaluation_file_path
    
    def evaluate_result(self, response: list | dict, label_data: list, chunk_dir: int, multi_agent: bool, r_lines: bool=False):
        '''
        single_llm: [
        ['+', 'filename', 'linenumber', 'linecontent', 'groupid']
        ]
        multi_agent: {
        'filename': [
        ['+', 'linenumber', 'groupid']
        ]
        }
        '''
        if multi_agent:
            try:
                r_list = []
                for key, item in response.items():
                    for line in item:
                        r_list.append([line[0], key, line[1], line[2]])
                response = r_list
            except Exception as e:
                print(e)
                print(f'error: {chunk_dir}')
                return 0.0
            
        groupid_index = 3 if multi_agent else 4

        
        model_lines = [tuple(item[:3]) for item in response] # ['+', 'filename', 'linenumber']
        correct_lines = []
        for label_entry in label_data:
            for line in label_entry['diff']:
                correct_lines.append(tuple(line[:3]))

        group_numbers = {item[groupid_index] for item in response}
        if len(group_numbers) < 2:
            if r_lines:
                return [0, len(model_lines)]
            return 0.0
        
        model_groups = []
        for group_num in group_numbers:
            group = [tuple(item[:3]) for item in response if item[groupid_index] == group_num]
            model_groups.append(group)
        
        correct_groups = []
        for label_entry in label_data:
            group = [tuple(line[:3]) for line in label_entry['diff']]
            correct_groups.append(group)
        
        weight_matrix = []
        for model_group in model_groups:
            model_set = set(model_group)
            row = []
            for correct_group in correct_groups:
                correct_set = set(correct_group)
                row.append(len(model_set & correct_set))
            weight_matrix.append(row)
        
        if not weight_matrix:
            if r_lines:
                return [0, len(model_lines)]
            return 0.0
        
        weight_matrix_np = np.array(weight_matrix, dtype=np.int64)
        cost_matrix = -weight_matrix_np 
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_matching = np.sum(weight_matrix_np[row_ind, col_ind])
        
        total_lines = len(model_lines)
        if r_lines:
            return [total_matching, total_lines]
        return total_matching / total_lines if total_lines != 0 else 0.0
    
    def get_single_diff(self, repo_path, chunk_dir):
        label_data =[]
        chunk_path = os.path.join(repo_path, chunk_dir)
        single_diff_path = os.path.join(chunk_path, "atom_commit")
        if os.path.exists(single_diff_path):
            for diff_file in os.listdir(single_diff_path):
                if diff_file.endswith(".json"):
                    with open(os.path.join(single_diff_path, diff_file), "r", encoding='utf-8') as f:
                        try:
                            label_data.append({
                                'diff': json.load(f)
                            })
                        except json.JSONDecodeError:
                            continue
        return label_data
    
    def call_mal(self) -> Dict:
        accuracy_dict = {}
        cbar = tqdm.tqdm(os.listdir(self.result_dir), desc="Evaluating results")
        for file in cbar:
            result_path = self.result_dir
            repo_name = file.split('_')[0]

            if not os.path.isdir(result_path):
                print(f"Result path {result_path} does not exist.")
                continue

            repo_path = os.path.join(self.base_dir, repo_name)
            if not os.path.isdir(repo_path):
                print(f"Repository path {repo_path} does not exist.")
                continue

            if not accuracy_dict.get(repo_name, None):
                accuracy_dict[repo_name] = []
            llm_response = {}
            
            try:
                with open(os.path.join(result_path, file), "r", encoding='utf-8') as f:
                    llm_response = json.load(f)
                    answer = llm_response.get('answer')
            except Exception as e:
                continue


            chunk_dir = file.split('_')[1].split(".")[0]
            label_data = self.get_single_diff(repo_path, chunk_dir)
            try:
                evaluation_result = self.evaluate_result(answer, label_data, chunk_dir, True)
            except Exception as e:
                evaluation_result = 0.0
            accuracy_dict[repo_name].append({
                'chunk_dir': chunk_dir,
                'accuracy': evaluation_result,
            }) 
        
        for repo_name, results in accuracy_dict.items():
            total = len(results)
            accuracy = sum([result['accuracy'] for result in results]) / total if total > 0 else 0.0
            accuracy_dict[repo_name] = {
                'accuracy': accuracy,
                'details': results,
            }

        overall_accuracy = sum([result['accuracy'] * result['num'] for result in accuracy_dict.values()]) / sum([result['num'] for result in accuracy_dict.values()]) if len(accuracy_dict) > 0 else 0.0
        accuracy_dict['overall_accuracy'] = overall_accuracy

        
        with open(self.evaluation_file_path, "w") as f:
            json.dump(accuracy_dict, f, indent=4)

        return accuracy_dict
    
def get_repo_accuracy(repo_filter, accuracy_type, e_result):
    r_list = []
    for repo_name, item in e_result.items():
        if isinstance(item, float):
            continue
        if repo_name in repo_filter:
            r_list.append({
                'accuracy': item.get(accuracy_type, 0.0),
                'num': item.get('num', 0),
            })
    total_num = sum([item['num'] for item in r_list])
    overall_accuracy = sum([item['accuracy'] * item['num'] for item in r_list]) / total_num if total_num > 0 else 0.0
    return overall_accuracy

def accuracy_distribution():
    ma_type ={
        '0_1_1_1_0': 'ColaUntangle',
        '0_0_1_1_0': 'ColaUntangle_nocomments',
        '0_1_0_0_0': 'w/o Info Tools',
        '5_1_1_1_0': 'GPT-o4-min',
        '1_1_1_1_0': 'GPT-4o',
        '4_1_1_1_0': 'Qwen',
        '2_1_1_1_0': 'Claude4-Sonet',
    }

    llm_type = {
        '0_1_1_1_0': 'LLM_ZeroShot (w/o Collaborative)',
        '0_1_1_1_1': 'LLM_CoT',
        '0_1_1_0_0': 'EA',
        '0_1_0_1_0': 'IA',
    }

    ma_data_dir = '../results/multi_agent'
    llm_data_dir = '../results/llm'

    config = Config()
    c_repo = config.c_repos
    j_repo = config.java_repos

    for key, value in ma_type.items():
        ma_type[key] = {
            'name': value,
            'accuracy': 0.0,
        }
    for key, value in llm_type.items():
        llm_type[key] = {
            'name': value,
            'accuracy': 0.0,
        }
    
    for key, value in ma_type.items():
        e_result = load_json(os.path.join(ma_data_dir, key, 'evaluation_result.json'))
        ma_type[key]['accuracy'] = e_result['overall_accuracy']
        ma_type[key]['accuracya'] = e_result.get('overall_accuracya', 0.0)
        ma_type[key]['c_accuracy'] = get_repo_accuracy(c_repo, 'accuracy', e_result)
        ma_type[key]['java_accuracy'] = get_repo_accuracy(j_repo, 'accuracy', e_result)
        ma_type[key]['c_accuracya'] = get_repo_accuracy(c_repo, 'accuracya', e_result)
        ma_type[key]['java_accuracya'] = get_repo_accuracy(j_repo, 'accuracya', e_result)

    for key, value in llm_type.items():
        e_result = load_json(os.path.join(llm_data_dir, key, 'evaluation_result.json'))
        llm_type[key]['accuracy'] = e_result['overall_accuracy']
        llm_type[key]['accuracya'] = e_result.get('overall_accuracya', 0.0)
        llm_type[key]['c_accuracy'] = get_repo_accuracy(c_repo, 'accuracy', e_result)
        llm_type[key]['java_accuracy'] = get_repo_accuracy(j_repo, 'accuracy', e_result)
        llm_type[key]['c_accuracya'] = get_repo_accuracy(c_repo, 'accuracya', e_result)
        llm_type[key]['java_accuracya'] = get_repo_accuracy(j_repo, 'accuracya', e_result)

    
    print(f'Accuracy All:')
    print("Multi-Agent Results:")
    for key, value in ma_type.items():
        print(f"{value['name']}:")
        print(f"C: {value['c_accuracy']:.2f}/{value['c_accuracya']:.2f} "
            f"Java: {value['java_accuracy']:.2f}/{value['java_accuracya']:.2f} "
            f"Overall: {value['accuracy']:.2f}/{value['accuracya']:.2f} ")

    print("\nLLM Results:")
    for key, value in llm_type.items():
        print(f"{value['name']}:")
        print(
            f"C: {value['c_accuracy']:.2f}/{value['c_accuracya']:.2f} "
            f"Java: {value['java_accuracy']:.2f}/{value['java_accuracya']:.2f} "
            f"Overall: {value['accuracy']:.2f}/{value['accuracya']:.2f} ")

    
if __name__ == "__main__":
    accuracy_distribution()