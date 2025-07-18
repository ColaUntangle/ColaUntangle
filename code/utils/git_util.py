import shutil
import subprocess
from typing import List


class Git_Util(object):
    def __init__(self, temp_dir: str = 'temp'):
        self.temp_dir = temp_dir

    def _clean_up(self):
        for path in self.temp_paths:
            shutil.rmtree(path, ignore_errors=True)

    def __enter__(self):
        self.temp_paths = []
        return self

    def __exit__(self, *exc_details):
        self._clean_up()


    @staticmethod
    def get_current_head(path: str) -> str:
        get_head_process = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                                            bufsize=1, stdout=subprocess.PIPE, cwd=path).stdout
        show_lines = list(map(lambda line: line.decode('utf-8', 'replace'), get_head_process.readlines()))
        get_head_process.close()

        return show_lines[0]

    @staticmethod
    def get_commit_msg(sha: str, path: str) -> str:
        commit_msg_process = subprocess.Popen(('git log --format=%B -n 1 ' + sha).split(' '),
                                              bufsize=1, stdout=subprocess.PIPE, cwd=path).stdout
        show_lines = list(map(lambda line: line.decode('utf-8', 'replace'), commit_msg_process.readlines()))
        commit_msg_process.close()

        return '\n'.join(show_lines)

    @staticmethod
    def get_all_commit_hashes(path: str) -> List[str]:
        commit_hashes_process = subprocess.Popen(['git', 'log', '--branches=*', '--format=oneline'], bufsize=1,
                                                 stdout=subprocess.PIPE, cwd=path).stdout
        commit_hashes = commit_hashes_process.readlines()
        commit_hashes_process.close()
        return list(map(lambda line: line.decode('utf-8', 'replace').split(' ')[0].strip(), commit_hashes))


    @staticmethod
    def get_file_content(sha: str, filepath: str, repo_path: str) -> str:
        try:
            cmd = ['git', 'show', f'{sha}:{filepath}']
            result = subprocess.run(cmd, cwd=repo_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            content = result.stdout.decode('utf-8', 'replace')
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            return content
        except subprocess.CalledProcessError:
            return ''

    