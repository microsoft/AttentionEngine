from itertools import product
import os
import json
from benchmark.bench_utils import bench_sigmoidattn_fwd, bench_attention_fwd

import concurrent.futures
import traceback

import torch
from tvm import tl
import tvm.tl.language as T

def cache_module(tuned_config, kernel, output_idx_list, problem_keys):
    try:# cache
        # mod = tl.cached(kernel, output_idx_list, BATCH, H, N_CTX, D_HEAD, D_HEADV, *tuned_config.values())
        program = kernel(*problem_keys.values(), *tuned_config.values())
        mod, params = tl.lower(program)
        return mod, params, tuned_config
    except Exception as e:
        print(traceback.format_exc())
        return None, None, tuned_config

def tl_tune(kernel, problem_keys, tuned_configs, output_idx_list=[4,], file_path="tuned_result.json"):

    # problem_keys = {
    #     "B": BATCH, "H": H, "N_CTX": N_CTX, "D_HEAD": D_HEAD, "D_HEADV": D_HEADV, "causal":True
    # }
    # cache
    if os.path.exists(file_path):
        with open(file_path,"r", encoding='utf-8') as file:
            data = json.load(file)
        for item in data:
            if all(item.get(key) == value for key, value in problem_keys.items()):
                tuned_config = item.get('tuned_config')
                return tuned_config, item.get('latency')
    else:
        with open(file_path, "w", encoding='utf-8') as file:
            json.dump([], file, ensure_ascii=False, indent=4)


        
    output_idx_list = output_idx_list# [4]
    best_latency = 1e6
    best_tflops = 0
    best_tflops_ref = 0
    best_config = None
    best_output_dict = {}
    latencys = []
    configs_out = []
    log_file = file_path.replace(".json",".log")
    log = open(log_file, "a")


    # Step 1: Cache modules in parallel
    cached_results = []

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(cache_module, config, kernel, output_idx_list, problem_keys): config for config in tuned_configs}

        for future in concurrent.futures.as_completed(futures):
            mod, params, tuned_config = future.result()
            if mod is not None:
                cached_results.append((mod, params, tuned_config))
    # print(cached_results)
    # Step 2: Benchmark serially
    for mod, params, tuned_config in cached_results:
        try:
            # mod = tl.cached(kernel, output_idx_list, BATCH, H, N_CTX, D_HEAD, D_HEADV, *tuned_config.values())
            # output_dict  = bench_func(mod, BATCH, H, N_CTX, D_HEAD, D_HEADV)
            mod_profile = tl.Profiler(mod, params, output_idx_list, tl.TensorSupplyType.Randn)
            latency = mod_profile.do_bench(mod_profile, n_warmup=5, n_repeat=10, profiler="torch")
            # latency = output_dict['latency']
            # tflops = output_dict['tflops']
            # ref_tflops = output_dict['tflops_ref']
            output_dict = {
                "latency": latency,
                # "tflops": tflops,
                # "tflops_ref": ref_tflops
            }
        except Exception as e:
            print(traceback.format_exc())
            latency = 1e6
            tflops = 0
            ref_tflops = 0
            output_dict = {}


        if latency < best_latency:
            best_latency = latency
            # best_tflops = tflops
            best_config = tuned_config
            # best_tflops_ref = ref_tflops
            best_output_dict = output_dict

        # print(latency, tflops, ref_tflops)
        print(output_dict)
        print(tuned_config)
        latencys.append(latency)
        configs_out.append(configs_out)
        log.write(f"Latency: {latency}, Config: {tuned_config}\n")
        log.flush()


    # for tuned_config in tuned_configs:
    #     try:
    #         mod = tl.cached(kernel, output_idx_list, BATCH, H, N_CTX, D_HEAD, D_HEADV, *tuned_config.values())
    #     except Exception as e:
    #         print(e)
    #         continue
    #     try:
    #         latency = bench_sigmoidattn_fwd(mod, BATCH, H, N_CTX, D_HEAD, D_HEADV)
    #     except Exception as e:
    #         print(e)
    #         latency = 1e6
    #     if latency < best_latency:
    #         best_latency = latency
    #         best_config = tuned_config
    #     print(latency)
    #     print(tuned_config)

    # append to file
    if True: # best_config is not None:
        new_entry = problem_keys.copy()
        new_entry['tuned_config'] = best_config
        # new_entry['latency'] = best_latency
        # new_entry['tflops'] = best_tflops
        # new_entry['ref_tflops'] = ref_tflops
        new_entry.update(best_output_dict)

        with open(file_path, "r+", encoding='utf-8') as file:
            data = json.load(file)
            data.append(new_entry)
            file.seek(0)
            json.dump(data, file, ensure_ascii=False, indent=4)

    return best_config, best_latency

    
class AttnFwdTunner:
    def __init__(self,DK, DV, block_M, block_N, num_threads, stages, **kwargs):
        self.DK = DK
        self.DV = DV
        self.block_M = block_M
        self.block_N = block_N
        self.num_threads = num_threads
        self.stages = stages


    def generate_config(self):
        block_M, block_N, num_threads, stages = self.block_M, self.block_N, self.num_threads, self.stages
        _configs = list(product(block_M, block_N, num_threads, stages))
        configs  = [
            {
                'block_M': c[0], 'block_N': c[1], 'stages': c[3], 'thread_num': c[2], 'shared_fuse': False
            }
            for c in _configs
        ]

        tuned_configs = []
        for c in configs:
            num_warps = c['thread_num'] // 32
            if c['block_M'] % (num_warps*16):
                continue

            tuned_configs.append(c)
                

        return tuned_configs

    
    
    def tl_tune(self, kernel, problem_keys, tuned_configs, output_idx_list=[4,], file_path="tuned_result.json"):

        return tl_tune(kernel, problem_keys, tuned_configs, output_idx_list, file_path)[0]