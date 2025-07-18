import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from utils.config_utils import Config
from utils.json_utils import load_json


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

def build_table(abbreviation_dict, language_type, remove_accuracya=False, row_name_map=None, base_result_dir=None):    
    table = {}
    overall_acc = {}
    c_repo = Config().c_repos
    j_repo = Config().java_repos
    for repo_name, repo_abbr in abbreviation_dict.items():
        table_row = {}
        for method_name, method_path in row_name_map.items():
            eval_file_path = os.path.join(base_result_dir, method_path, 'evaluation_result.json')
            eval_data = load_json(eval_file_path)
            if repo_name in eval_data:
                repo_eval = eval_data[repo_name]
                accuracy = '{:.0f}'.format(repo_eval.get('accuracy', 0.0) * 100)
                accuracya = '{:.0f}'.format(repo_eval.get('accuracya', 0.0) * 100) if 'accuracya' in repo_eval else 'None'
                if remove_accuracya or accuracya == '' or accuracya == 'None':
                    value = accuracy
                else:
                    value = f"{accuracy}/{accuracya}"
            table_row[method_name] = value
            if remove_accuracya:
                overall_acc[method_name] = f'{get_repo_accuracy(j_repo, "accuracy", eval_data) * 100:.0f}'
            else:
                overall_acc[method_name] = f'{get_repo_accuracy(c_repo, "accuracy", eval_data) * 100:.0f}/{get_repo_accuracy(c_repo, "accuracya", eval_data) * 100:.0f}'
        table[repo_abbr] = table_row

    df = pd.DataFrame.from_dict(table, orient='index')
    df.index.name = f'{language_type}_repo'

    overall_row = {}
    for col in df.columns:
        overall_row[col] = overall_acc.get(col, 'N/A')
    df.loc['Overall'] = overall_row
    df = df.transpose()

    return df


def RQ1_overall_performance():
    print('[RQ1] Overall Performance of C# and Java Repos: ')
    config = Config()
    c_abbreviation = config.c_repo_abbreviation
    j_abbreviation = config.java_repo_abbreviation
    base_result_dir = '../results/'

    row_name_map = {
        'LLMZeroShot': 'llm/0_1_1_1_0',
        'LLMCoT': 'llm/0_1_1_1_1',
        'ColaUntangleNoComment': 'multi_agent/0_0_1_1_0',
        'ColaUntangle': 'multi_agent/0_1_1_1_0',
    }

    result_dir = "../results/tables"
    os.makedirs(result_dir, exist_ok=True)

    c_table_df = build_table(c_abbreviation, 'c#', row_name_map=row_name_map, base_result_dir=base_result_dir)
    c_table_df.to_csv(os.path.join(result_dir, 'c#_overall_performance.csv'))

    java_table_df = build_table(j_abbreviation, 'java', remove_accuracya=True, row_name_map=row_name_map, base_result_dir=base_result_dir)
    java_table_df.to_csv(os.path.join(result_dir, 'java_overall_performance.csv'))

    print("✅ Tables generated at ../results/tables/\n")

def RQ1_round_1_consensus():
    print('[RQ1] rounds distribution and consensus rate in round 1: ')
    path = '../results/multi_agent/0_1_1_1_0/untangled_results/'
    if not os.path.exists(path):
        print(f"❌ {path} does not exist, please run download results first!\n")
        return
    cbar = tqdm(os.listdir(path), desc="Processing files")
    all, same_1 = 0, 0

    def normalize_answer(answer):
        result = {}
        for filename, items in answer.items():
            group_map = defaultdict(set)
            for entry in items:
                sign, line, group = entry
                group_map[group].add((sign, line))
            result[filename] = sorted([frozenset(v) for v in group_map.values()])
        return result

    def compare_answers(ex_answer, im_answer):
        ex_norm = normalize_answer(ex_answer)
        im_norm = normalize_answer(im_answer)
        if set(ex_norm.keys()) != set(im_norm.keys()):
            return False
        for fname in ex_norm:
            if sorted(ex_norm[fname]) != sorted(im_norm[fname]):
                return False
        return True
    
    round_list = []
    for file in cbar:
        if not file.endswith('.json'):
            continue
        file_path = os.path.join(path, file)
        try:
            data = load_json(file_path)
            if 'case_history' in data:
                opinions = data['case_history'].get('rounds', [])[0].get('opinions', [])
                if len(opinions) < 2:
                    continue
                round = data['case_history'].get('total_rounds', 1)
                ex_answer = opinions[0].get('opinion', {}).get('answer', {})
                im_answer = opinions[1].get('opinion', {}).get('answer', {})
                round_list.append(round)
                same_grouping = compare_answers(ex_answer, im_answer)
                if same_grouping and round == 1:
                    same_1 += 1
                all += 1
        except Exception as e:
            pass
    
    print(f'✅ Average Round: {sum(round_list) / len(round_list):.4f} 1_Round_Rate: {round_list.count(1) / all:.4f} 2_Round_Rate: {round_list.count(2) / all:.4f}, 3_Round_Rate: {round_list.count(3) / all:.4f}')
    print(f"✅ The ratio between explicit and implicit worker agents who do not reach consensus during the initial round: {(all - same_1) / all if all > 0 else 0:.4f}\n")

def RQ1_stmts_acc():
    print('[RQ1] Stmts and Accuracies of C and Java Repos: ')
    os.makedirs('../results/figs', exist_ok=True)

    node_num_path = '../data/corpora_raw/stmt_num.json'
    if not os.path.exists(node_num_path):
        print(f"❌ {node_num_path} does not exist, please run download dataset first!")
        return
    merge_dict = load_json(node_num_path).get('comments')
    evaluation_result_path = '../results/multi_agent/0_1_1_1_0/evaluation_result.json'
    evaluation_result = load_json(evaluation_result_path)

    c_repos = Config().c_repos
    j_repos = Config().java_repos

    def calculate_metrics(repos):
        stmt_avg = {}
        acc_avg = {}
        for repo_name in repos:
            stmt_values = list(merge_dict.get(repo_name, {}).values())
            stmt_avg[repo_name] = sum(stmt_values) / len(stmt_values) if stmt_values else 0
            
            repo_eval = evaluation_result.get(repo_name, {})
            if isinstance(repo_eval, float):
                acc_values = [repo_eval]
            else:
                details = repo_eval.get('details', [])
                acc_values = [chunk.get('accuracy', 0) for chunk in details]
            acc_avg[repo_name] = sum(acc_values) / len(acc_values) if acc_values else 0
        return stmt_avg, acc_avg

    c_stmt_avg, c_acc_avg = calculate_metrics(c_repos)
    j_stmt_avg, j_acc_avg = calculate_metrics(j_repos)

    def sort_repos(stmt_avg, acc_avg):
        sorted_names = sorted(acc_avg.keys(), key=lambda x: acc_avg[x], reverse=True)
        return sorted_names, [stmt_avg[name] for name in sorted_names], [acc_avg[name] for name in sorted_names]

    c_sorted_names, c_sorted_stmt, c_sorted_acc = sort_repos(c_stmt_avg, c_acc_avg)
    j_sorted_names, j_sorted_stmt, j_sorted_acc = sort_repos(j_stmt_avg, j_acc_avg)

    font_config = {
        'size': 30
    }
    plt.rc('font', **font_config)
    plt.rcParams['pdf.use14corefonts'] = True

    c_repo_abbreviation = {
        'Commandline': 'CL',
        'CommonMark.NET': 'CM',
        'Hangfire': 'HF',
        'Humanizer': 'HU',
        'Lean': 'LE',
        'Nancy': 'NA',
        'Newtonsoft.Json': 'NJ',
        'Ninject': 'NI',
        'RestSharp': 'RS',
    }

    java_repo_abbreviation = {
        'spring-boot': 'SB',
        'elasticsearch': 'ES',
        'RxJava': 'RJ',
        'guava': 'GU',
        'retrofit': 'RE',
        'dubbo': 'DU',
        'ghidra': 'GH',
        'zxing': 'ZX',
        'druid': 'DR',
        'EventBus': 'EB',
    }

    def plot_combined_chart(repo_names, stmt_values, acc_values, filename):
        fig, ax1 = plt.subplots(figsize=(10 if filename == 'c_stmt_acc' else 12, 8))
        x = np.arange(len(repo_names))
        width = 0.35

        color_acc = '#BDBAD2'
        bars_acc = ax1.bar(x - width/2, acc_values, width, color=color_acc, label='Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='y')
        ax1.set_ylim(0, 1.05)
        ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
        
        ax2 = ax1.twinx()
        color_stmt = '#E88C81'
        bars_stmt = ax2.bar(x + width/2, stmt_values, width, color=color_stmt, label='# Stmts')
        ax2.set_ylabel('# Stmts')
        ax2.tick_params(axis='y')
        ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
        
        # Set statement axis limits
        max_stmt = max(stmt_values)
        ax2.set_ylim(0, int(max_stmt * 1.2) + 1)
        
        abb_repo_names = []
        for name in repo_names:
            if name in c_repo_abbreviation:
                abb_repo_names.append(c_repo_abbreviation[name])
            elif name in java_repo_abbreviation:
                abb_repo_names.append(java_repo_abbreviation[name])
            else:
                abb_repo_names.append(name)
        ax1.set_xticks(x)
        ax1.set_xticklabels(abb_repo_names)
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        
        plt.tight_layout()
        plt.savefig(f'../results/figs/{filename}.pdf', 
                   bbox_inches='tight', dpi=300, format='pdf', 
                   metadata={'CreationDate': None})
        plt.close()

    plot_combined_chart(c_sorted_names, c_sorted_stmt, c_sorted_acc, 'c_stmt_acc')
    plot_combined_chart(j_sorted_names, j_sorted_stmt, j_sorted_acc, 'java_stmt_acc')

    print(f'✅ Figures are saved in /results/figs/ successfully!\n')

def RQ2_ablation_study():
    print('[RQ2] Ablation Study of C# and Java Repos: ')
    row_name_map = {
        'w/o Info Tools': 'multi_agent/0_1_0_0_0',
        'w/o Collaborative': 'llm/0_1_1_1_0',
        'w/o W1 w/o Collaborative': 'llm/0_1_0_1_0',
        'w/o W2 w/o Collaborative': 'llm/0_1_1_0_0',
        'ColaUntangle': 'multi_agent/0_1_1_1_0',
    }
    config = Config()
    c_repo = config.c_repos
    j_repo = config.java_repos

    results = []

    for row_name, relative_path in row_name_map.items():
        full_path = os.path.join('../results', relative_path, 'evaluation_result.json')
        e_result = load_json(full_path)

        overall_acc = e_result.get('overall_accuracy', 0.0) * 100
        overall_acca = e_result.get('overall_accuracya', 0.0) * 100

        c_acc = get_repo_accuracy(c_repo, 'accuracy', e_result) * 100
        c_acca = get_repo_accuracy(c_repo, 'accuracya', e_result) * 100

        java_acc = get_repo_accuracy(j_repo, 'accuracy', e_result) * 100
        java_acca = get_repo_accuracy(j_repo, 'accuracya', e_result) * 100

        overall_cell = f'{overall_acc:.0f}/{overall_acca:.0f}' if overall_acca else f'{overall_acc:.0f}'
        c_cell = f'{c_acc:.0f}/{c_acca:.0f}'
        java_cell = f'{java_acc:.0f}/{java_acca:.0f}'

        results.append({
            'Method': row_name,
            'C# Dataset': c_cell,
            'Java Dataset': java_cell,
            'Overall': overall_cell,
        })


    df = pd.DataFrame(results)
    df = df[['Method', 'C# Dataset', 'Java Dataset', 'Overall']] 

    output_dir = '../results/tables'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'ablation_study.csv')
    df.to_csv(output_path, index=False)

    print("✅ Tables generated at ../results/tables/\n")

def RQ3_acc_of_models():
    print('[RQ3] Accuracies of Different Models: ')
    row_name_map = {
        'Deepseek-V3': 'multi_agent/0_1_1_1_0',
        'ChatGPT-4o-mini': 'multi_agent/5_1_1_1_0',
        'Chatgpt-4o': 'multi_agent/1_1_1_1_0',
        'Qwen3-235b': 'multi_agent/4_1_1_1_0',
        'Claude-4-Sonet': 'multi_agent/2_1_1_1_0'
    }

    config = Config()
    c_repo = config.c_repos
    j_repo = config.java_repos

    results = []
    for row_name, relative_path in row_name_map.items():
        full_path = os.path.join('../results', relative_path, 'evaluation_result.json')
        e_result = load_json(full_path)

        overall_acc = e_result.get('overall_accuracy', 0.0) * 100
        overall_acca = e_result.get('overall_accuracya', 0.0) * 100

        c_acc = get_repo_accuracy(c_repo, 'accuracy', e_result) * 100
        c_acca = get_repo_accuracy(c_repo, 'accuracya', e_result) * 100

        java_acc = get_repo_accuracy(j_repo, 'accuracy', e_result) * 100
        java_acca = get_repo_accuracy(j_repo, 'accuracya', e_result) * 100

        overall_cell = f'{overall_acc:.0f}/{overall_acca:.0f}' if overall_acca else f'{overall_acc:.0f}'
        c_cell = f'{c_acc:.0f}/{c_acca:.0f}'
        java_cell = f'{java_acc:.0f}/{java_acca:.0f}'

        results.append({
            'Model': row_name,
            'C# Dataset': c_cell,
            'Java Dataset': java_cell,
            'Overall': overall_cell,
        })


    df = pd.DataFrame(results)
    df = df[['Model', 'C# Dataset', 'Java Dataset', 'Overall']] 

    output_dir = '../results/tables'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'model_sensitivity.csv')
    df.to_csv(output_path, index=False)

    print("✅ Tables generated at ../results/tables/\n")

def RQ3_round():
    print('[RQ3] Average rounds for different models: ')
    path_type = {
        'ChatGPT-4o-mini': '5_1_1_1_0',
        'Chatgpt-4o': '1_1_1_1_0',
        'Qwen3-235b': '4_1_1_1_0',
        'Claude-4-Sonet': '2_1_1_1_0',
        'ColaUtangle(Deepseek-V3)': '0_1_1_1_0',
    }

    for model, p in path_type.items():
        path = '../results/multi_agent/' + p + '/untangled_results/'
        if not os.path.exists(path):
            print(f"❌ {path} does not exist, please run download results first!\n")
            return
        round_list = []
        for file in os.listdir(path):
            if not file.endswith('.json'):
                continue
            file_path = os.path.join(path, file)
            try:
                data = load_json(file_path)
                if 'case_history' in data and 'processing_time' in data['case_history']:
                    round = data['case_history'].get('total_rounds', 1)
                    round_list.append(round)
                else:
                    pass
            except Exception as e:
                pass
        print(f'{model}: {sum(round_list) / len(round_list) if round_list else 0:.2f}')
    print('✅ Analysis completed!\n')

if __name__ == '__main__':
    RQ1_overall_performance()
    RQ1_round_1_consensus()
    RQ1_stmts_acc()
    RQ2_ablation_study()
    RQ3_acc_of_models()
    RQ3_round()


