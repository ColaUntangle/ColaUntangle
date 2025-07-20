import os
import sys
import tqdm
import networkx as nx

from utils.config_utils import Config
from utils.pdg_utils import  ReadUtils, ProcessUtils
      
            
def generate_pdg_neighborhoods(chunk_path, pdg_hop=1, context_path_dir=None):
    pdg_dir = os.path.join(chunk_path, 'explicit_contexts')
    if not os.path.exists(pdg_dir):
        print(f'Directory {pdg_dir} does not exist.')
        return
    merged_delta_pdg_path = os.path.join(pdg_dir, 'merged_deltaPDG.dot')
    filtered_delta_pdg_path = os.path.join(context_path_dir, f'implicit_contexts.dot')
    if os.path.exists(filtered_delta_pdg_path):
        return
    if not os.path.exists(merged_delta_pdg_path):
        return
    merged_delta_pdg = ReadUtils.obj_dict_to_networkx(ReadUtils.read_graph_from_dot(merged_delta_pdg_path) )

    filtered_delta_pdg = ProcessUtils.get_context_graph(merged_delta_pdg, hop=pdg_hop)
    if not os.path.exists(context_path_dir):
        os.makedirs(context_path_dir, exist_ok=True)
    nx.drawing.nx_pydot.write_dot(filtered_delta_pdg, filtered_delta_pdg_path)
    


def generate_contexts(base_data_dir, pdg_hop=1):
    for repo_name in os.listdir(base_data_dir):
        repo_path = os.path.join(base_data_dir, repo_name)
        if not os.path.isdir(repo_path):
            print(f"Repository path {repo_path} does not exist.")
            continue

        repo_path_bar = tqdm.tqdm(os.listdir(repo_path), desc=f"Processing {repo_path}", unit="chunk")
        for chunk_dir in repo_path_bar:
            repo_path_bar.set_postfix({'chunk_dir': chunk_dir})
            chunk_path = os.path.join(repo_path, chunk_dir)
            if not os.path.isdir(chunk_path):
                print(f"Chunk path {chunk_path} does not exist.")
                continue

            context_path_dir = os.path.join(chunk_path, 'implicit_contexts')
            # GENERATE PDG NEIGHBORHOODS
            generate_pdg_neighborhoods(chunk_path, pdg_hop=pdg_hop, context_path_dir=context_path_dir)
            continue
    return

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('To use this script please run as `[python] create_implicit_contexts.py '
              '<add_comments>')
        exit(1)
    
    add_comments = bool(int(sys.argv[1]))
    config = Config(add_comments=add_comments)
    base_data_dir = config.base_data_dir
    pdg_hop = 1

    generate_contexts(base_data_dir, pdg_hop=pdg_hop)
    
    