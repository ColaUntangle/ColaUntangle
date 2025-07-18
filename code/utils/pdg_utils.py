from rapidfuzz import fuzz
from typing import Tuple, Dict, List
import time
import itertools
import os
import json

import networkx as nx
import pydot
import re
import pygraphviz
from contextlib import contextmanager

@contextmanager
def timer(name=""):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f"{name} 操作耗时: {elapsed_time:.4f} 秒")

class Eq_Utils:
    def __init__(self, m_fuzziness, n_fuzziness):
        self.m_fuzziness = m_fuzziness
        self.n_fuzziness = n_fuzziness

    def context_eq(self, context_a: str, context_b: str) -> bool:
        return fuzz.ratio(context_a, context_b, score_cutoff=self.m_fuzziness) > 0

    def node_label_eq(self, node_label_a: str, node_label_b: str) -> bool:
        return fuzz.ratio(node_label_a, node_label_b, score_cutoff=self.n_fuzziness) > 0

    def node_eq(self, graph_a, node_a, graph_b, node_b):
        if not (self.node_label_eq(graph_a.nodes[node_a]['label'], graph_b.nodes[node_b]['label'])):
            return False

        n_a = [n for n in list(graph_a.successors(node_a)) + list(graph_a.predecessors(node_a)) if
               'color' not in graph_a.nodes[n].keys() or graph_a.nodes[n]['color'] == 'orange']

        n_b = [n for n in list(graph_b.successors(node_b)) + list(graph_b.predecessors(node_b)) if
               'color' not in graph_b.nodes[n].keys() or graph_b.nodes[n]['color'] == 'orange']

        # We check for set inclusion, make sure we have the smaller set in the outer loop!
        if len(n_a) > len(n_b):
            temp = n_b
            n_b = n_a
            n_a = temp

        for node in n_a:
            found = False
            for other_node in n_b:
                try:
                    if self.node_label_eq(graph_a.nodes[node]['label'], graph_b.nodes[other_node]['label']):
                        found = True
                        break
                except KeyError:
                    pass
            if not found:
                return False

        return True

    def attr_eq(self, attr_a, attr_b):
        for key in attr_a.keys():
            if key not in attr_b.keys():
                return False
            elif key == 'label' and not (self.node_label_eq(attr_a[key], attr_b[key])):
                return False
            elif attr_a[key] != attr_b[key]:
                return False
        return True

class ReadUtils:
    def __init__(self):
        pass

    def read_graph_from_dot(file_: str) -> Tuple[Dict, Dict[str, str]]:
        try:
            with open(file_, "r", encoding="utf-8") as f:
                dot_data = f.read()
            apdg = pydot.graph_from_dot_data(dot_data)[0].obj_dict
        except Exception as e:
            print(f"Error reading graph from {file_}: {e}")
            apdg = ""
        return apdg


    def obj_dict_to_networkx(obj_dict):
        graph = nx.MultiDiGraph()

        if isinstance(obj_dict, str):
            return graph

        for node, data in list(obj_dict['nodes'].items()):
            if node != 'graph' and 'span' in data[0]['attributes'].keys():
                attr = {k: v[1:-1] if v[0] == v[-1] == '"' else v for k, v in data[0]['attributes'].items()}
                graph.add_node(node, **attr)

        for edge, data in list(obj_dict['edges'].items()):
            s, t = edge
            attr = {k: v[1:-1] if v[0] == v[-1] == '"' else v for k, v in data[0]['attributes'].items()}
            graph.add_edge(s, t, **attr)

        for subgraph in list(obj_dict['subgraphs'].keys()):
            for node, data in list(obj_dict['subgraphs'][subgraph][0]['nodes'].items()):
                if node != 'graph' and 'span' in data[0]['attributes'].keys():
                    attr = {k: v[1:-1] if v[0] == v[-1] == '"' else v for k, v in data[0]['attributes'].items()}
                    if 'label' in obj_dict['subgraphs'][subgraph][0]['attributes'].keys():
                        attr['cluster'] = obj_dict['subgraphs'][subgraph][0]['attributes']['label'][1:-1]
                    elif 'graph' in obj_dict['subgraphs'][subgraph][0]['nodes'].keys():
                        attr['cluster'] = obj_dict['subgraphs'][subgraph][0]['nodes']['graph'][0]['attributes']['label'][1:-1]
                    graph.add_node(node, **attr)

        return graph

class ProcessUtils:
    def __init__(self):
        pass
    
    def trans_java_pdg(pdg_path):
        pdg = pygraphviz.AGraph(pdg_path)
        label_pattern = re.compile(r'^\s*(\d+)\s*:\s*(.+)$')

        def convert_node_name(old_name):
            if old_name.startswith('v') and old_name[1:].isdigit():
                return 'n' + old_name[1:]
            return old_name

        def process_nodes(graph):
            for node in graph.nodes():
                label = node.attr.get('label', '')
                match = label_pattern.match(label.strip('"'))
                if match:
                    line_number_str, statement = match.groups()
                    line_number = int(line_number_str.strip())
                    span_start = line_number - 1
                    span_end = line_number - 1
                    span_value = f"{span_start}-{span_end}"
                    node.attr['label'] = statement.strip()
                    node.attr['span'] = span_value
                else:
                    node.attr['span'] = '0-0'
                    node.attr['label'] = label.strip('"')

        process_nodes(pdg)
        for subgraph in pdg.subgraphs():
            process_nodes(subgraph)


        new_pdg = pygraphviz.AGraph(strict=False, directed=True)
        for node in pdg.nodes():
            old_name = node.get_name()
            new_name = convert_node_name(old_name)
            new_pdg.add_node(new_name, **dict(node.attr))

        def copy_subgraphs(src_graph, dst_graph):
            for subgraph in src_graph.subgraphs():
                new_subgraph = dst_graph.add_subgraph(
                    name=subgraph.name,
                    label=subgraph.attr.get("label", "")
                )
                for node in subgraph.nodes():
                    old_name = node.get_name()
                    new_name = convert_node_name(old_name)
                    new_subgraph.add_node(new_name, **dict(node.attr))
                copy_subgraphs(subgraph, new_subgraph)

        copy_subgraphs(pdg, new_pdg)

        for edge in pdg.edges():
            src = edge[0]
            dst = edge[1]
            new_src = convert_node_name(src)
            new_dst = convert_node_name(dst)
            new_pdg.add_edge(new_src, new_dst, **dict(edge.attr))

        new_pdg.write(pdg_path)
    
    def filter_and_reduce_graph(G: nx.MultiDiGraph, hop: int) -> nx.MultiDiGraph:
        target_nodes = {n for n, d in G.nodes(data=True) if d.get('color') in {'red', 'green'}}

        neighbor_nodes = set()
        for node in target_nodes:
            for depth in range(1, hop + 1):
                neighbors = nx.descendants_at_distance(G, node, distance=depth)
                neighbor_nodes.update(neighbors)

        kept_nodes = target_nodes | neighbor_nodes

        subG = G.subgraph(kept_nodes).copy()
        resultG = nx.MultiDiGraph()
        
        for n in subG.nodes:
            resultG.add_node(n, **subG.nodes[n])

        for source in target_nodes:
            for target in target_nodes:
                if source == target:
                    continue
                if nx.has_path(subG, source, target):
                    for path in nx.all_simple_paths(subG, source, target):
                        resultG.add_edge(source, target, label='shortcut', inferred=True)
                        break 
        return resultG
    
    def get_context_graph(G: nx.MultiDiGraph, hop: int) -> nx.MultiDiGraph:
        target_nodes = {n for n, d in G.nodes(data=True) if d.get('color') in {'red', 'green'}}
        neighbor_nodes = set()
        for node in target_nodes:
            for depth in range(1, hop + 1):
                neighbors = nx.descendants_at_distance(G, node, distance=depth)
                neighbor_nodes.update(neighbors)

        kept_nodes = target_nodes | neighbor_nodes
        resultG = nx.MultiDiGraph()
        for n in kept_nodes:
            resultG.add_node(n, **G.nodes[n])

        for u, v, key, data in G.edges(keys=True, data=True):
            if u in kept_nodes and v in kept_nodes:
                resultG.add_edge(u, v, key=key, **data)

        return resultG
    def nx_to_str(graph: nx.MultiDiGraph) -> str:
        graph_dict = nx.node_link_data(graph)
        graph_json_str = json.dumps(graph_dict, ensure_ascii=False)
        return graph_json_str
    

class Marked_Merger:
    def __init__(self, m_fuzziness, n_fuzziness):
        self.eq_utils = Eq_Utils(m_fuzziness, n_fuzziness)

    def __call__(self, before_apdg, after_apdg):
        before_apdg = before_apdg.copy()
        after_apdg = after_apdg.copy()
        label_map_ab = dict()
        label_map_ba = dict()

        for node, data in before_apdg.nodes(data=True):
            if 'color' in data.keys() and data['color'] != 'orange':
                continue
            for other_node, other_data in after_apdg.nodes(data=True):
                if other_node in label_map_ba.keys():
                    continue
                if 'color' in other_data.keys() and other_data['color'] != 'orange':
                    continue
                equivalent = self.eq_utils.node_eq(before_apdg, node, after_apdg, other_node)
                try:
                    equivalent = equivalent \
                                 and self.eq_utils.context_eq(data['cluster'], other_data['cluster'])
                except KeyError:
                    equivalent = equivalent \
                                 and 'cluster' not in data.keys() \
                                 and 'cluster' not in other_data.keys()
                if equivalent:
                    label_map_ab[str(node)] = str(other_node)
                    label_map_ba[str(other_node)] = str(node)
                    break
        # Visit anchors, explore neighbourhood and copy over nodes
        # Each node copied: add to a list to be explored
        # As each node is explored add copied nodes
        # Stop when list is empty
        # Boot strap list with all marked nodes in v2
        to_visit = [str(node) for node in after_apdg.nodes() if
                    'color' in after_apdg.nodes[node].keys() and after_apdg.nodes[node]['color'] != 'orange']
        visited = list()

        work_to_be_done = len(to_visit) > 0
        # counter = 1
        # We fixed-point compute this due to the fact that we leave potentially dangling edges, 
        # so we iterate until all edges point to real nodes
        while work_to_be_done:
            node_id = to_visit[0]
            node_id = node_id.replace('n', 'd') if node_id not in label_map_ba.keys() else label_map_ba[node_id]
            to_visit = to_visit[1:]
            if not (before_apdg.has_node(node_id) and 'label' in before_apdg.nodes[node_id].keys()):
                # Find node in after graph and visit if not visited (sanity check)
                other_node = 'n' + node_id[1:]
                if other_node not in visited:
                    before_apdg.add_node(node_id, **after_apdg.nodes[other_node])

                    # Add in-edges from after graph to before graph, we leave references to un-imported nodes dangling
                    # However, we also add the un-imported nodes to the to-visit list
                    in_edges = after_apdg.in_edges(nbunch=[other_node], keys=True)
                    for in_edge in list(in_edges):
                        in_, _, key = in_edge
                        if in_ not in label_map_ba.keys():
                            in_id = str(in_).replace('n', 'd')
                            if in_ not in to_visit and in_ not in visited: to_visit.append(in_)
                        else:
                            in_id = label_map_ba[str(in_)]
                        if before_apdg.has_edge(in_id, node_id, key):
                            if key in before_apdg[in_id][node_id]  and key in after_apdg[in_][other_node] and not self.eq_utils.attr_eq(before_apdg[in_id][node_id][key], after_apdg[in_][other_node][key]):
                                before_apdg.add_edge(in_id, node_id, key,
                                                     **after_apdg[in_][other_node][key])
                                after_apdg.remove_edge(in_, other_node, key)
                        else:
                            before_apdg.add_edge(in_id, node_id, key, **after_apdg[in_][other_node][key])
                            after_apdg.remove_edge(in_, other_node, key)

                    # Add out-edges from after graph to before graph, we leave references to un-imported nodes dangling
                    # However, we also add the un-imported nodes to the to-visit list
                    out_edges = after_apdg.out_edges(nbunch=[other_node], keys=True)
                    for out_edge in list(out_edges):
                        _, out_, key = out_edge
                        if out_ not in label_map_ba.keys():
                            out_id = str(out_).replace('n', 'd')
                            if out_ not in to_visit and out_ not in visited: to_visit.append(out_)
                        else:
                            out_id = label_map_ba[str(out_)]
                        if before_apdg.has_edge(node_id, out_id, key):
                            if key in before_apdg[node_id][out_id] and key in after_apdg[other_node][out_] and not self.eq_utils.attr_eq(before_apdg[node_id][out_id][key], after_apdg[other_node][out_][key]):
                                before_apdg.add_edge(node_id, out_id, key,
                                                     **after_apdg[other_node][out_][key])
                                after_apdg.remove_edge(other_node, out_, key)
                        else:
                            before_apdg.add_edge(node_id, out_id, key,
                                                 **after_apdg[other_node][out_][key])
                            after_apdg.remove_edge(other_node, out_, key)
                    visited.append(other_node)
            work_to_be_done = len(to_visit) > 0

        for node in before_apdg.nodes():
            if 'color' in before_apdg.nodes[node].keys():
                for edge in list(before_apdg.in_edges(nbunch=[node], keys=True)) \
                            + list(before_apdg.out_edges(nbunch=[node], keys=True)):
                    s, t, k = edge
                    before_apdg[s][t][k]['color'] = before_apdg.nodes[node]['color']

        for node, other_node in label_map_ab.items():
            for edge in list(after_apdg.in_edges(nbunch=[other_node], keys=True)):
                source, sink, key = edge
                assert sink == other_node
                if source in label_map_ba.keys():
                    before_node = label_map_ba[source]
                    if before_apdg.has_edge(before_node, node):
                        if key in before_apdg[before_node][node] and key in after_apdg[source][sink] and not self.eq_utils.attr_eq(before_apdg[before_node][node][key], after_apdg[source][sink][key]):
                            after_apdg[source][sink][key]['color'] = 'green'
                            before_apdg.add_edge(before_node, node, key, **after_apdg[source][sink][key])
                            after_apdg.remove_edge(source, sink, key)
                    else:
                        after_apdg[source][sink][key]['color'] = 'green'
                        before_apdg.add_edge(before_node, node, key, **after_apdg[source][sink][key])
                        after_apdg.remove_edge(source, sink, key)

            for edge in list(after_apdg.out_edges(nbunch=[other_node], keys=True)):
                source, sink, key = edge
                assert source == other_node
                if sink in label_map_ba.keys():
                    before_node = label_map_ba[sink]
                    if before_apdg.has_edge(node, before_node, key):
                        if key in before_apdg[node][before_node] and key in after_apdg[source][sink] and not self.eq_utils.attr_eq(before_apdg[node][before_node][key], after_apdg[source][sink][key]):
                            after_apdg[source][sink][key]['color'] = 'green'
                            before_apdg.add_edge(node, before_node, key, **after_apdg[source][sink][key])
                            after_apdg.remove_edge(source, sink, key)
                    else:
                        after_apdg[source][sink][key]['color'] = 'green'
                        before_apdg.add_edge(node, before_node, key, **after_apdg[source][sink][key])
                        after_apdg.remove_edge(source, sink, key)

        for edge in list(after_apdg.edges(keys=True)):
            source, target, key = edge
            if after_apdg[source][target][key]['style'] != 'solid':
                after_apdg.remove_edge(source, target, key=key)

        for node, other_node in label_map_ab.items():
            for edge in before_apdg.out_edges(nbunch=[node], data=True, keys=True):
                source, sink, key, data = edge
                if data['style'] != 'solid':
                    continue
                assert source == node
                if 'color' not in data.keys():
                    if sink in label_map_ab.keys():
                        after_node = label_map_ab[sink]
                        if not (after_apdg.has_edge(other_node, after_node)):
                            before_apdg[source][sink][key]['color'] = 'red'

        return before_apdg

# merge the deltaPDGs of the same commit into a single graph
class DeltaPDGsMerger:
    def __init__(self):
        pass

    @staticmethod
    def find_entry_and_exit(context, graph):
        # This works under an assumption of entry and exit uniqueness
        # See the change in marking nodes to ensure this: Not marking Entry and Exit nodes
        entry = None
        exit = None
        for node, data in graph.nodes(data=True):
            if 'label' in data.keys() and 'cluster' in data.keys():
                if 'Entry' in data['label'] and data['cluster'] == context:
                    entry = node
                elif 'Exit' in data['label'] and data['cluster'] == context:
                    exit = node

        return entry, exit

    @staticmethod
    def get_context_from_nxgraph(graph):
        contexts = dict()
        for node in graph.nodes():
            if 'cluster' in list(graph.nodes[node].keys()):
                contexts[str(node)] = graph.nodes[node]['cluster']
        return contexts

    @staticmethod
    def merge_deltas_for_a_commit(delta_pdgs: list):
        # We will take the first graph as a base and add the rest onto it
        if len(delta_pdgs) == 0:
            return None
        graph = delta_pdgs[0]['graph']
        original_file = delta_pdgs[0]['location']
        contexts = DeltaPDGsMerger.get_context_from_nxgraph(graph)
        output = graph.copy()
        delta_pdgs = delta_pdgs[1:]
        for i, delta_pdg in enumerate(delta_pdgs):
            next_graph = delta_pdg['graph']
            graph_location = delta_pdg['location']
            next_contexts = DeltaPDGsMerger.get_context_from_nxgraph(next_graph)
            # First find the contexts that exist in both
            mappable_contexts = list()
            for next_context, current_context in itertools.product(set(next_contexts.values()), set(contexts.values())):
                if next_context == current_context and next_context != 'lambda expression':
                    mappable_contexts.append(current_context)
                    break

            copied_nodes = list()
            mapped_nodes = list()
            # And copy over all of the nodes into the merged representation
            for context in mappable_contexts:
                current_entry, current_exit = DeltaPDGsMerger.find_entry_and_exit(context, graph)
                other_entry, other_exit = DeltaPDGsMerger.find_entry_and_exit(context, next_graph)

                if current_entry is not None and other_entry is not None:
                    mapped_nodes.append((str(current_entry), str(other_entry)))
                if current_exit is not None and other_exit is not None:
                    mapped_nodes.append((str(current_exit), str(other_exit)))

                other_nodes = [n for n in next_graph.nodes(data=True)
                            if n[0] not in [other_entry, other_exit] and 'cluster' in n[1].keys()
                            and n[1]['cluster'] == context]
                if current_entry is None and other_entry is not None:
                    other_nodes.append((other_entry, next_graph.node[other_entry]))
                if current_exit is None and other_exit is not None:
                    other_nodes.append((other_exit, next_graph.node[other_exit]))

                if len(other_nodes) > 0:
                    if current_entry is not None and 'file' not in graph.nodes[current_entry].keys():
                        graph.nodes[current_entry]['file'] = os.path.basename(graph_location[:-len('.dot')])
                    if current_exit is not None and 'file' not in graph.nodes[current_exit]:
                        graph.nodes[current_exit]['file'] = os.path.basename(graph_location[:-len('.dot')])

                for copy_node, data in other_nodes:
                    data['file'] = os.path.basename(graph_location[:-len('.dot')])
                    output.add_node('m%d_' % i + copy_node, **data)
                    copied_nodes.append(('m%d_' % i + copy_node, copy_node))

            # Now we copy over all of the contexts that did not map/exist in the merged representation
            for other_context in [c for c in set(next_contexts.values()) if c not in mappable_contexts]:
                other_entry, other_exit = DeltaPDGsMerger.find_entry_and_exit(other_context, next_graph)
                other_nodes = [n for n in next_graph.nodes(data=True)
                            if n[0] not in [other_entry, other_exit] and 'cluster' in n[1].keys()
                            and n[1]['cluster'] == other_context]
                # For aesthetic reasons make sure to copy entry first and exit last
                if other_entry is not None:
                    other_nodes = [(other_entry, next_graph.nodes[other_entry])] + other_nodes
                if other_exit is not None:
                    other_nodes.append((other_exit, next_graph.nodes[other_exit]))
                for copy_node, data in other_nodes:
                    data['file'] = os.path.basename(graph_location[:-len('.dot')])
                    output.add_node('m%d_' % i + copy_node, **data)
                    copied_nodes.append(('m%d_' % i + copy_node, copy_node))

            # Finally we copy over all of the nodes w/o a context
            for copy_node, data in [n for n in next_graph.nodes(data=True) if n[0] not in next_contexts.keys()]:
                data['file'] = os.path.basename(graph_location[:-len('.dot')])
                output.add_node('m%d_' % i + copy_node, **data)
                copied_nodes.append(('m%d_' % i + copy_node, copy_node))

            # We move over the edges making sure we properly map the ends
            reverse_map = {v: u for u, v in copied_nodes + mapped_nodes}
            for copied_node, original_node in copied_nodes:
                for s, t, k in next_graph.edges(nbunch=[original_node], keys=True):
                    try:
                        if s in reverse_map.keys() and t in reverse_map.keys():
                            if output.has_node(reverse_map[s]) and output.has_node(reverse_map[t]):
                                output.add_edge(reverse_map[s], reverse_map[t], key=k, **next_graph[s][t][k])
                    except KeyError:
                        pass

        # And finally we mark the original file nodes
        for node, _ in [n for n in output.nodes(data=True) if 'file' not in n[1].keys()]:
            graph.nodes[node]['file'] = original_file

        return output

class DeltaPDGGenerator:
    def __init__(self, before_pdg_location: str, after_pdg_location: str, m_fuzziness: int = 100, n_fuzziness: int = 100):
        self.before_pdg = ReadUtils.obj_dict_to_networkx(ReadUtils.read_graph_from_dot(before_pdg_location))
        self.after_pdg = ReadUtils.obj_dict_to_networkx(ReadUtils.read_graph_from_dot(after_pdg_location))

        self.merger = Marked_Merger(m_fuzziness=m_fuzziness, n_fuzziness=n_fuzziness)

    def __call__(self, diff: List[Tuple[str, str, int, str]]):
        
        marked_before = self.mark_pdg_nodes(self.before_pdg, '-', diff)
        marked_after = self.mark_pdg_nodes(self.after_pdg, '+', diff)
        
        self.deltaPDG = self.merger(before_apdg=marked_before, after_apdg=marked_after)
            
        return self.deltaPDG
    
    
    def mark_pdg_nodes(self, apdg, marker: str,
                   diff: List[Tuple[str, str, int, str]]) -> pygraphviz.AGraph:
        marked_pdg = apdg.copy()
        change_label = 'green' if marker == '+' else 'red'
        anchor_label = 'orange'
        c_diff = [ln for m, f, ln, line in diff if m == marker]
        for node, data in marked_pdg.nodes(data=True):
            if data['label'] in ['Entry', 'Exit']:
                attr = data
                attr['label'] += ' %s' % data['cluster']
                apdg.add_node(node, **attr)
                continue  # Do not mark entry and exit nodes.
            try:
                start, end = [int(ln) for ln in data['span'].split('-') if '-' in data['span']]
                start, end = start + 1, end + 1  # Adjust for 1-based indexing
                data['span'] = f"{start}-{end}"
            except ValueError:
                continue
            # We will use the changed nodes as anchors via neighbours
            change = any([start <= cln <= end for cln in c_diff])
            # anchor = any([start <= aln - 1 <= end for aln in a_diff])
            if change:
                attr = data
                attr['color'] = change_label if change else anchor_label
                apdg.add_node(node, **attr)

        return marked_pdg

    