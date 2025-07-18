import json
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import networkx as nx

from utils.git_util import Git_Util
from tqdm import tqdm
from utils.config_utils import Config
from utils.git_util import Git_Util
from utils.pdg_utils import DeltaPDGGenerator, DeltaPDGsMerger, ReadUtils, ProcessUtils
import threading

thread_name_to_index = {}
thread_index_counter = [1]  
thread_name_lock = threading.Lock()

def get_thread_index_csharp():
    thread_name = threading.current_thread().name
    with thread_name_lock:
        if thread_name not in thread_name_to_index:
            thread_name_to_index[thread_name] = thread_index_counter[0]
            thread_index_counter[0] += 1
        return thread_name_to_index[thread_name]

def get_thread_index_java():
    pid = os.getpid()
    return pid
    
def clear_thread_config():
    global thread_name_to_index
    global thread_index_counter
    with thread_name_lock:
        thread_name_to_index = {}
        thread_index_counter = [1]


def clearDir(path):
    if os.path.exists(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                clearDir(file_path)
                os.rmdir(file_path)


def generate_pdg_per_file_java(file_path: str, git_repo_path: str, head_sha: str, end_sha: str, diff: list, extractor_path: str, result_dir: str, thread_id: int, timeout: int = 20) -> None:
    """
    Generate PDG for a given file (JAVA) with a timeout.
    """
    initial_content = ''
    final_content = ''
    file_path = file_path[1:] if file_path.startswith('/') else file_path
    temp_dir = f'../temp/java_pdg_temp/{thread_id}'
    os.makedirs(temp_dir, exist_ok=True)
    try:
        initial_content = Git_Util.get_file_content(head_sha + '^', file_path, git_repo_path)
    except FileNotFoundError:
        print(f"File {file_path} not found in the repository.")
    
    try:
        final_content = Git_Util.get_file_content(end_sha, file_path, git_repo_path)
    except FileNotFoundError:
        print(f"File {file_path} not found in the repository.")

    temp_file = 'temp.java' 
    generate_dot_filename = 'temp-PDG-DATA.dot'
    before_pdg_file, after_pdg_file = 'before.dot', 'after.dot'
    extractor_dir = os.path.dirname(extractor_path)
    popen_args = {
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
        'text': True,
        'bufsize': 1,
        'cwd': os.path.normpath(os.path.join(os.getcwd(), extractor_dir))
    }

    try:
        with open(os.path.join(temp_dir, temp_file), 'w', encoding='utf-8') as f:
            f.write(initial_content)
        cmd_args = [
            "java",
            "-jar",
            "progex.jar",
            "-outdir",
            os.path.normpath(os.path.join(os.getcwd(), temp_dir)),
            "-pdg",
            os.path.normpath(os.path.join(os.getcwd(), temp_dir, temp_file))
        ]
        proc = subprocess.Popen(cmd_args, **popen_args)
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Timeout: Analysis for {file_path} (before) took too long.")
            proc.kill()
            proc.communicate()  # clean up
            return nx.MultiDiGraph()

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        copy_dot_file_and_remove_src(os.path.join(temp_dir, generate_dot_filename), os.path.join(result_dir, before_pdg_file))
        ProcessUtils.trans_java_pdg(os.path.join(result_dir, before_pdg_file))

        with open(os.path.join(temp_dir, temp_file), 'w', encoding='utf-8') as f:
            f.write(final_content)
        cmd_args = [
            "java",
            "-jar",
            "progex.jar",
            "-outdir",
            os.path.normpath(os.path.join(os.getcwd(), temp_dir)),
            "-pdg",
            os.path.normpath(os.path.join(os.getcwd(), temp_dir, temp_file))
        ]
        proc = subprocess.Popen(cmd_args, **popen_args)
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Timeout: Analysis for {file_path} (after) took too long.")
            proc.kill()
            proc.communicate()
            return nx.MultiDiGraph()

        copy_dot_file_and_remove_src(os.path.join(temp_dir, generate_dot_filename), os.path.join(result_dir, after_pdg_file))
        ProcessUtils.trans_java_pdg(os.path.join(result_dir, after_pdg_file))

    finally:
        if os.path.exists(os.path.join(temp_dir, temp_file)):
            os.remove(os.path.join(temp_dir, temp_file))
    
    try:
        deltaPDGGenerator = DeltaPDGGenerator(os.path.join(result_dir, before_pdg_file), os.path.join(result_dir, after_pdg_file))
        deltaPDG = deltaPDGGenerator(diff)
    except Exception as e:
        print(f"Delta PDG is empty for {result_dir}.")
        return nx.MultiDiGraph()

    if len(deltaPDG.nodes) == 0:
        print(f"Delta PDG is empty for {result_dir}.")
        return nx.MultiDiGraph()

    delta_path = os.path.join(result_dir, 'deltaPDG.dot')
    nx.drawing.nx_pydot.write_dot(deltaPDG, delta_path)
    return deltaPDG

def copy_dot_file_and_remove_src(src_path: str, dst_path: str) -> None:
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        os.remove(src_path)
    else:
        with open(dst_path, 'w', encoding='utf-8') as f:
            f.write('digraph empty { }')  

def generate_pdg_per_file_csharp(
    file_path: str, git_repo_path: str, head_sha: str, end_sha: str,
    diff: list, extractor_path: str, result_dir: str,
    timeout: int = 20
) -> nx.MultiDiGraph:
    """
    Generate PDG for a given file (C#) with a timeout.
    """
    initial_content = ''
    final_content = ''
    file_path = file_path[1:] if file_path.startswith('/') else file_path
    try:
        initial_content = Git_Util.get_file_content(head_sha + '^', file_path, git_repo_path)
    except FileNotFoundError:
        print(f"File {file_path} not found in the repository.")
    
    try:
        final_content = Git_Util.get_file_content(end_sha, file_path, git_repo_path)
    except FileNotFoundError:
        print(f"File {file_path} not found in the repository.")

    temp_file = 'temp.cs' 
    generate_dot_filename = 'pdg.dot'
    before_pdg_file, after_pdg_file = 'before.dot', 'after.dot'
    extractor_dir = os.path.dirname(extractor_path)
    popen_args = {
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
        'text': True,
        'bufsize': 1,
        'cwd': os.path.normpath(os.path.join(os.getcwd(), extractor_dir))
    }

    try:
        with open(os.path.join(extractor_dir, temp_file), 'w', encoding='utf-8') as f:
            f.write(initial_content)
        cmd_args = [extractor_path, '.', '.\\' + temp_file]
        proc = subprocess.Popen(cmd_args, **popen_args)
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()  # clean up
            return nx.MultiDiGraph()

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        copy_dot_file_and_remove_src(os.path.join(extractor_dir, generate_dot_filename), os.path.join(result_dir, before_pdg_file))


        with open(os.path.join(extractor_dir, temp_file), 'w', encoding='utf-8') as f:
            f.write(final_content)
        cmd_args = [extractor_path, '.', '.\\' + temp_file]
        proc = subprocess.Popen(cmd_args, **popen_args)
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return nx.MultiDiGraph()

        copy_dot_file_and_remove_src(os.path.join(extractor_dir, generate_dot_filename), os.path.join(result_dir, after_pdg_file))
    finally:
        if os.path.exists(os.path.join(extractor_dir, temp_file)):
            os.remove(os.path.join(extractor_dir, temp_file))
    
    deltaPDGGenerator = DeltaPDGGenerator(os.path.join(result_dir, before_pdg_file), os.path.join(result_dir, after_pdg_file))
    deltaPDG = deltaPDGGenerator(diff)
    if len(deltaPDG.nodes) == 0:
        print(f"Delta PDG is empty for {result_dir}.")
        return nx.MultiDiGraph()

    delta_path = os.path.join(result_dir, 'deltaPDG.dot')
    nx.drawing.nx_pydot.write_dot(deltaPDG, delta_path)
    return deltaPDG


def process_chunk_dir(chunk_dir: str, repo_path: str, git_repo_path: str, repo_name: str, language: str, extractor_path: str, pdg_hop: int) -> None:
    """Processes a single chunk directory."""
    if language == 'csharp':
        thread_id = get_thread_index_csharp()
    elif language == 'java':
        thread_id = get_thread_index_java()
    print(f'Thread {thread_id} repo {repo_name}: Processing chunk directory: {chunk_dir}')
    chunk_path = os.path.join(repo_path, chunk_dir)
    merged_diffs_dir = os.path.join(chunk_path, 'merged_diffs')

    if not os.path.exists(merged_diffs_dir) or len(os.listdir(merged_diffs_dir)) != 1:
        print(f"Thread {thread_id} repo {repo_name}: Skipping {chunk_dir} due to missing or incorrect merged_diffs directory.")
        return

    pdg_dir = os.path.join(chunk_path, 'pdgs')
    if not os.path.exists(pdg_dir):
        os.makedirs(pdg_dir)

    merged_delta_pdg_path = os.path.join(pdg_dir, 'merged_deltaPDG.dot')
    filtered_delta_pdg_path = os.path.join(pdg_dir, f'filtered_merged_deltaPDG_{pdg_hop}.dot')
    if os.path.exists(merged_delta_pdg_path) and os.path.exists(filtered_delta_pdg_path):
        print(f"Thread {thread_id} repo {repo_name}: Skipping {chunk_dir} as PDGs already exist.")
        return

    merged_diff_file = os.listdir(merged_diffs_dir)[0]
    merge_content = []
    try:
        with open(os.path.join(merged_diffs_dir, merged_diff_file), 'r') as f:
            merge_content = json.load(f)
    except json.JSONDecodeError:
        print(f"Thread {thread_id} repo {repo_name}: Error decoding JSON from {merged_diff_file} in {chunk_dir}.")
        return


    head_sha = merged_diff_file.split('_')[0] 
    end_sha = merged_diff_file.split('_')[-1].split('.')[0]

    modified_files = {change[1] for change in merge_content}
    file_delta_list = []
    
    extractor_path_thread = extractor_path.replace("THREADNUM", str(thread_id)) 
    file_name_index = 0
    for file_path in modified_files:
        print(f"Thread {thread_id} repo {repo_name}: chunk_dir: {chunk_dir}  progress: {file_name_index}/{len(modified_files)}")
        result_file_dir = os.path.join(pdg_dir, f'{file_name_index}') 
        if os.path.exists(os.path.join(result_file_dir, 'deltaPDG.dot')):
            try:
                single_file_delta_pdg = ReadUtils.obj_dict_to_networkx(ReadUtils.read_graph_from_dot(os.path.join(result_file_dir, 'deltaPDG.dot')))
                file_delta_list.append({
                    'graph': single_file_delta_pdg,
                    'location': file_path + f'--{file_name_index}.dot',
                })
                file_name_index += 1
                continue
            except Exception as e:
                pass
        diff = [line for line in merge_content if line[1] == file_path]
        if not os.path.exists(result_file_dir):
            os.makedirs(result_file_dir)

        single_file_delta_pdg = None
        if language == 'csharp':
            single_file_delta_pdg = generate_pdg_per_file_csharp(file_path, git_repo_path, head_sha, end_sha, diff, extractor_path_thread, result_file_dir, 10)
        elif language == 'java':
            single_file_delta_pdg = generate_pdg_per_file_java(file_path, git_repo_path, head_sha, end_sha, diff, extractor_path_thread, result_file_dir, int(thread_id), 10)

        if single_file_delta_pdg is not None:
            file_delta_list.append({
                'graph': single_file_delta_pdg,
                'location': file_path + f'--{file_name_index}.dot',
            })
        file_name_index += 1

    merged_delta_pdg = DeltaPDGsMerger.merge_deltas_for_a_commit(file_delta_list)
    nx.drawing.nx_pydot.write_dot(merged_delta_pdg, merged_delta_pdg_path)

    filtered_delta_pdg = ProcessUtils.filter_and_reduce_graph(merged_delta_pdg, hop=pdg_hop)
    nx.drawing.nx_pydot.write_dot(filtered_delta_pdg, filtered_delta_pdg_path)

    context_path_dir = os.path.join(chunk_path, 'contexts')
    os.makedirs(context_path_dir, exist_ok=True)
    neighbor_delta_pdg_path = os.path.join(context_path_dir, f'context_deltaPDG_{pdg_hop}.dot')
    neighbor_delta_pdg = ProcessUtils.get_context_graph(merged_delta_pdg, hop=1)
    nx.drawing.nx_pydot.write_dot(neighbor_delta_pdg, neighbor_delta_pdg_path)


def generate_pdg_per_repo_csharp(repo_path: str, git_repo_path: str, repo_name: str, language: str, add_comments: bool,
                         extractor_path: str, pdg_hop: int = 0, num_threads: int = 4) -> None:

    chunk_dirs = os.listdir(repo_path)
    total = len(chunk_dirs)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        pbar = tqdm(total=total, desc=f'Processing {repo_name}', position=0)

        for chunk_dir in chunk_dirs:
            future = executor.submit(
                process_chunk_dir,
                chunk_dir, repo_path, git_repo_path, repo_name, language,
                add_comments, extractor_path, pdg_hop
            )
            future.add_done_callback(lambda p: pbar.update(1))
            futures.append(future)

        for future in as_completed(futures):
            future.result()
        pbar.close()

def generate_pdg_per_repo_java(repo_path: str, git_repo_path: str, repo_name: str, language: str, add_comments: bool,
                         extractor_path: str, pdg_hop: int = 0) -> None:
    chunk_dirs = os.listdir(repo_path)

    pbar = tqdm(chunk_dirs, desc=f'Processing {repo_name}', position=0)

    for chunk_dir in pbar:
        try:
            process_chunk_dir(chunk_dir, repo_path, git_repo_path, repo_name, language,
                add_comments, extractor_path, pdg_hop)
        except Exception as e:
            continue

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('To use this script please run as `[python] create_explicit_contexts.py '
              '<add_comments> <language>')
        exit(1)
    
    add_comments = bool(int(sys.argv[1]))
    language = sys.argv[2]

    if language not in ['java', 'csharp']:
        print('language should be java or csharp')
        exit(1)


    config = Config(add_comments=add_comments)

    hop = config.pdg_hop
    base_data_dir = config.base_data_dir
    subjects_dir = config.subjects_dir
    extractor_path = config.csharp_extractor_path if language == 'csharp' else config.java_extractor_path
    thread_num = config.pdg_thread_num

    for repo_name in os.listdir(base_data_dir):
        print(f'Processing {repo_name}...')
        git_repo_name = config.git_path_name_map[repo_name]
        data_path = os.path.join(base_data_dir, repo_name)
        git_repo_path = os.path.join(subjects_dir, git_repo_name)
        if not os.path.exists(data_path) or not os.path.exists(git_repo_path):
            print(f"Repository {repo_name} does not exist. Skipping...")
            continue
        if language == 'csharp':
            clear_thread_config()
        if repo_name in config.c_repos:
            generate_pdg_per_repo_csharp(
                repo_path=data_path,
                git_repo_path=git_repo_path,
                repo_name=repo_name,
                language=language,
                add_comments=add_comments,
                extractor_path=extractor_path,
                pdg_hop=hop,
                num_threads=thread_num, 
            )
        elif repo_name in config.java_repos:
            generate_pdg_per_repo_java(
                repo_path=data_path,
                git_repo_path=git_repo_path,
                repo_name=repo_name,
                language=language,
                add_comments=add_comments,
                extractor_path=extractor_path,
                pdg_hop=hop
            )
        
        


