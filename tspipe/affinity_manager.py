import os
import re
import subprocess
from copy import deepcopy
from typing import Dict, List, Optional

import tspipe.multiprocessing as mp


class AffinityManager():
    def __init__(self, process: Optional[mp.Process] = None):
        self.process = process
        self.pid = process.pid if process is not None else os.getpid()

    @staticmethod
    def parse_affinity(affinity: str) -> List[int]:
        assert re.match(r"^(\d+(-\d+)?)(,\d+(-\d+)?)*$", affinity)

        result = []
        for chunk in affinity.split(","):
            if '-' in chunk:
                left, right = map(int, chunk.split('-'))
                for i in range(left, right + 1):
                    result.append(i)
            else:
                result.append(int(chunk))
        return result

    @staticmethod
    def num_cpus_for_gpu(gpu_id: int, affinities: Dict) -> int:
        cpu_map = {cpu: 0 for cpu in affinities[gpu_id]}
        for aff_lst in affinities.values():
            for cpu in aff_lst:
                if cpu in cpu_map:
                    cpu_map[cpu] += 1
        max_gpu_per_cpu = max(cpu_map.values())
        return min(len(affinities[gpu_id]) // max_gpu_per_cpu, 4)

    @staticmethod
    def get_gpu_affinity_map() -> Dict[int, List[int]]:
        gpu_affinities: Dict[int, List[int]] = {}
        nvidia_smi_proc = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True)
        affinity_col_idx = None
        for line_idx, line in enumerate(nvidia_smi_proc.stdout.decode("utf-8").split('\n')):
            if line_idx == 0:
                for idx, colname in enumerate(line.split('\t')):
                    if 'CPU Affinity' in colname:
                        affinity_col_idx = idx

            if line.startswith('GPU'):
                cols = line.split('\t')
                gid = int(cols[0].replace('GPU', '').strip())
                aff = AffinityManager.parse_affinity(cols[affinity_col_idx].strip())
                gpu_affinities[gid] = aff

        gpu_affinities_ = deepcopy(gpu_affinities)

        for gid in sorted(gpu_affinities.keys()):
            gpu_demand = AffinityManager.num_cpus_for_gpu(gid, gpu_affinities_)
            assert len(gpu_affinities[gid]) >= gpu_demand
            chosen_affinity = gpu_affinities[gid][:gpu_demand]

            for other_gid in gpu_affinities.keys():
                if gid != other_gid:
                    for af in chosen_affinity:
                        if af in gpu_affinities[other_gid]:
                            gpu_affinities[other_gid].remove(af)

            gpu_affinities[gid] = chosen_affinity

        return gpu_affinities

    def set_affinity(self, cpu_list: List[int]) -> None:
        cmdline = ["taskset", "-cp", ','.join(str(x) for x in cpu_list), str(self.pid)]
        subprocess.run(cmdline, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    def set_affinity_for_gpu(self, gpu_id: int):
        gpu_affinities = AffinityManager.get_gpu_affinity_map()
        self.set_affinity(gpu_affinities[gpu_id])

    def set_affinity_for_scheduler(self):
        gpu_affinities = AffinityManager.get_gpu_affinity_map()
        cpu_remainder = sorted(list(set(range(os.cpu_count())) - set(c for v in gpu_affinities.values() for c in v)))
        self.set_affinity(cpu_remainder)
