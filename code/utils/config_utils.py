from typing import Union
class Config:
    def __init__(self, add_comments: bool=False, add_graphs: bool = False, add_file_content: bool = False, add_concern_num: Union[bool, int] = 0, model: dict = None, ma_model_category: str = 'deepseek'):
        if add_comments:
            self.base_data_dir = '../data/corpora_raw/1/'
        else:
            self.base_data_dir = '../data/corpora_raw/0/'
        
        self.ma_model_category_map = {
            'deepseek-v3': '0',
            'chatgpt-4o': '1',
            'claude4-sonet': '2',
            'qwen': '4',
            'o4-mini': '5'
        }

        self.c_repo_abbreviation = {
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

        self.java_repo_abbreviation = {
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
        
            
        self.base_data_no_comments_dir = '../data/corpora_raw/0/'
        self.base_data_with_comments_dir = '../data/corpora_raw/1/'

        self.base_ma_result_dir = f"../results/multi_agent/{self.ma_model_category_map.get(ma_model_category)}_{int(add_comments)}_{int(add_graphs)}_{int(add_file_content)}_{int(add_concern_num)}/untangled_results/"
        self.base_ma_clean_result_dir = f"../results/multi_agent/{self.ma_model_category_map.get(ma_model_category)}_{int(add_comments)}_{int(add_graphs)}_{int(add_file_content)}_{int(add_concern_num)}/clean_untangled_results/"

        self.base_result_dir = f"../results/llm/{self.ma_model_category_map.get(ma_model_category)}_{int(add_comments)}_{int(add_graphs)}_{int(add_file_content)}_{int(add_concern_num)}/untangled_results/"
        self.base_clean_result_dir = f"../results/llm/{self.ma_model_category_map.get(ma_model_category)}_{int(add_comments)}_{int(add_graphs)}_{int(add_file_content)}_{int(add_concern_num)}/clean_untangled_results/"

        self.llm_repo_filter = ['Commandline', 'Hangfire', 'Lean', 'Nancy', 'CommonMark.NET', 'Newtonsoft.Json', 'Ninject', 'RestSharp', 'Humanizer', 'spring-boot', 'elasticsearch', 'RxJava',  'retrofit', 'dubbo', 'ghidra', 'zxing', 'druid', 'EventBus', 'guava']

        self.ma_repo_filter =['Commandline', 'Hangfire', 'Lean', 'Nancy', 'CommonMark.NET', 'Newtonsoft.Json', 'Ninject', 'RestSharp', 'Humanizer', 'spring-boot', 'elasticsearch', 'RxJava',  'retrofit', 'dubbo', 'ghidra', 'zxing', 'druid', 'EventBus', 'guava']

        self.subjects_dir = '../subjects/'

        self.git_path_name_map = {
            'Commandline': 'commandline',
            'Hangfire': 'Hangfire',
            "Lean": "Lean",
            "Nancy": "Nancy",
            "CommonMark.NET": "CommonMark.NET",
            "Newtonsoft.Json": "Newtonsoft.Json",
            "Ninject": "Ninject",
            "RestSharp": "RestSharp",
            'spring-boot': 'spring-boot',
            'elasticsearch': 'elasticsearch',
            'RxJava': 'RxJava',
            'guava': 'guava',
            'retrofit': 'retrofit',
            'dubbo': 'dubbo',
            'ghidra': 'ghidra',
            'zxing': 'zxing',
            'druid': 'druid',
            'EventBus': 'EventBus',
            'Humanizer': 'Humanizer'
        }

    
        self.c_repos = ['Commandline', 'Hangfire', 'Lean', 'Nancy', 'CommonMark.NET', 'Newtonsoft.Json', 'Ninject', 'RestSharp', 'Humanizer']
        self.java_repos  = ['guava', 'spring-boot', 'elasticsearch', 'RxJava',  'retrofit', 'dubbo', 'ghidra', 'zxing', 'druid', 'EventBus']

        self.java_extractor_path = '../tools/JavaExtractors/PROGEX.jar'
        self.csharp_extractor_path = '../tools/CsharpExtractors/execute_THREADNUM/PdgExtractor.exe'
        self.pdg_hop = 0
        self.pdg_thread_num = 6

    
    def get_opposite_result_dir(self):
        add_comments = int(self.base_result_dir.split('/')[3].split('_')[0])
        add_graphs = int(self.base_result_dir.split('/')[3].split('_')[1])
        add_file_content = int(self.base_result_dir.split('/')[3].split('_')[2])
        add_concern_num = int(self.base_result_dir.split('/')[3].split('_')[3])
        return f"../results/llm/{1-add_comments}_{add_graphs}_{add_file_content}_{add_concern_num}/untangled_results/"
