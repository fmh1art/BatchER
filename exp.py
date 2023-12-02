from src.prompting import ERDataset, GPTPOOL, Prompt, SinglePrompt, Item_type
from src.helper import get_fea_of_entries, save_json, open_json, tokens_num, cal_f1, mkdir, logger_config
from src.model import EncoderModel
from random import sample, shuffle
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import time
import random
import copy
import tiktoken
import traceback
import datetime


class Experiment:
    def __init__(self, gpt: GPTPOOL, data: ERDataset, dataset_name: str, batch_size: int, num_batches=-1,
                 EXP='HowBatching', cluster_percentile=1, sim_type='StructureAware', sim_func='ratio'):
        self.dataset_name = data.dataset_name
        self.task_description = f'When determining whether two {Prompt.Item_type}s are the same, you should only focus on critical properties and overlook noisy factors.\n'
        self.fea_sim_mode = 'PROP_BASED'
        self.sim_func_name = 'ratio'

        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.gpt = gpt
        self.data = data
        self.batch_size = batch_size
        self.num_batches = num_batches

        self.save_root = f'{EXP}'
        mkdir(f'./{self.save_root}')
        self.logger = logger_config(
            log_path=f'./{self.save_root}/log.txt', logging_name=f'{EXP}EXP')

        if dataset_name in ['em-beer', 'em-fz', 'em-ia', 'em-ag', 'em-wa']:
            min_samples = 1
        else:
            min_samples = self.batch_size

        self.encoder = EncoderModel(
            sim_func) if 'bert' in sim_func.lower() else None

        # if prop is null, then sim val is 0
        self.pair_fea_arrs = self._extact_fea_arrs(
            self.data.test_t1, self.data.test_t2, self.dataset_name, sim_type=sim_type, sim=sim_func, encoder=self.encoder)
        self.demo_feas_arrs = self._extact_fea_arrs(
            self.data.D_t1, self.data.D_t2, self.dataset_name, sim_type=sim_type, sim=sim_func, encoder=self.encoder)
        self.logger.info(
            f'pair_fea_arrs shape: {self.pair_fea_arrs.shape}, demo_feas_arrs shape: {self.demo_feas_arrs.shape}')

        self.dis = self._dis_of_pairdemos_Eucli(
            self.pair_fea_arrs, self.demo_feas_arrs)
        self.dis_threshold = np.percentile(self.dis, cluster_percentile)
        self.logger.info(f'dis_threshold: {self.dis_threshold}')

        self.dbscan = DBSCAN(eps=self.dis_threshold, min_samples=min_samples)

        self.preds = self.dbscan.fit_predict(self.pair_fea_arrs)
        self.logger.info(f'cluster num: {max(self.preds)+1}')
        self.p2c, self.c2p = self._generate_2dicts(self.preds)
        self.weight_p, self.weight_d = self._get_weights(self.data)

    def _get_weights(self, data):
        weight_p = [tokens_num(self.encoding, a+b)+5 for a,
                    b in zip(data.test_t1, data.test_t2)]
        weight_d = [tokens_num(self.encoding, a+b)+5 for a,
                    b in zip(data.D_t1, data.D_t2)]
        return weight_p, weight_d

    def save_prompt(self, batches, batch_type='similar', demo4batches=None, affx=None, explain=False, data=None):
        if data is None:
            data = self.data
        if affx is None:
            affx = 'fewshot' if demo4batches is not None else ''
        tol_ask = []
        self.logger.info(
            f'============================= querying {batch_type} batch =============================')
        for batch_id, batch in enumerate(batches):
            self.logger.info(
                f'------------------- query batch [{batch_id+1}/{self.num_batches}] of {batch_type} batch -------------------')

            pairs = [[data.test_t1[i], data.test_t2[i], data.test_lab[i]]
                     for i in batch]

            if demo4batches is not None:
                demos = demo4batches[batch_id]
                demos = [[data.D_t1[i], data.D_t2[i], data.D_lab[i]]
                         for i in demos]
                ask = Prompt.generate_prompt_for_gourp(
                    self.task_description, demos, pairs, explain, remove_noval_keys=False, dataset_name=data.dataset_name)
            else:
                ask = Prompt.generate_prompt_for_zeroshot_package(
                    self.task_description, pairs)

            tol_ask.append(ask)
            save_json(
                tol_ask, f'./{self.save_root}/[{batch_type}_batch]{affx}_tol_ask.json')

    def query_batch(self, batches, batch_type='similar', demo4batches=None, affx=None, explain=False, data=None):
        if data is None:
            data = self.data
        if affx is None:
            affx = 'fewshot' if demo4batches is not None else ''
        fail2reply, false_reply, tol_records = [], [], []
        self.logger.info(
            f'============================= querying {batch_type} batch =============================')
        for batch_id, batch in enumerate(batches):
            self.logger.info(
                f'------------------- query batch [{batch_id+1}/{self.num_batches}] of {batch_type} batch -------------------')

            pairs = [[data.test_t1[i], data.test_t2[i], data.test_lab[i]]
                     for i in batch]

            if demo4batches is not None:
                demos = demo4batches[batch_id]
                demos = [[data.D_t1[i], data.D_t2[i], data.D_lab[i]]
                         for i in demos]
                ask = Prompt.generate_prompt_for_gourp(
                    self.task_description, demos, pairs, explain, remove_noval_keys=False, dataset_name=data.dataset_name)
            else:
                ask = Prompt.generate_prompt_for_zeroshot_package(
                    self.task_description, pairs)

            query_cnt = 0
            max_cnt = 5
            while True:
                if query_cnt > max_cnt:
                    break
                try:
                    self.logger.info(ask)

                    # 计算错误的数量
                    tol, err, fn, fp = 0, 0, 0, 0
                    for r in tol_records:
                        tol += len(r['preds'])
                        err += sum([1 if p != l else 0 for p,
                                   l in zip(r['preds'], r['labs'])])
                        fn += sum([1 if p == 0 and l == 1 else 0 for p,
                                  l in zip(r['preds'], r['labs'])])
                        fp += sum([1 if p == 1 and l == 0 else 0 for p,
                                  l in zip(r['preds'], r['labs'])])
                    self.logger.info(
                        f'------------------- {err}/{tol}, fp: {fp}, fn: {fn}-------------------')

                    ans = self.gpt.query(ask, get_lower=False)
                    self.logger.info(ans)
                except Exception as e:
                    self.logger.info(
                        '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! err! [GPT fails to replay] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    err_info = str(traceback.format_exc())
                    if "This model's maximum context length" in err_info:
                        query_cnt += 1
                    self.logger.info(err_info)
                    record = {
                        'ask': ask,
                        'batch_id': batch_id,
                        'batch': batch,
                        'ask_tok': tokens_num(self.encoding, ask),
                    }
                    if demo4batches is not None:
                        record['demos'] = demos
                    fail2reply.append(record)
                    save_json(
                        fail2reply, f'./{self.save_root}/[{batch_type}_batch]{affx}_fail2reply.json')
                    time.sleep(random.randint(8, 10))
                    continue

                ans = ans.replace('No', 'no').replace('Yes', 'yes').replace(
                    'Question', 'question').replace('QUESTION', 'question')+'.'
                yf = ans.count('yes.') + ans.count('yes,')
                nf = ans.count('no.') + ans.count('no,')
                try:
                    assert yf+nf == len(batch)
                except Exception as e:
                    self.logger.info(
                        f'!!!!!!!!!!!!!!!!!!!!!!!!!!!! err! [yf: {yf}, nf: {nf} wrong numbers] !!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    self.logger.info(traceback.format_exc())
                    record = {
                        'ask': ask,
                        'ans': ans,
                        'batch_id': batch_id,
                        'batch': batch,
                        'yf': yf,
                        'nf': nf,
                        'ask_tok': tokens_num(self.encoding, ask),
                        'ans_tok': tokens_num(self.encoding, ans),
                    }
                    if demo4batches is not None:
                        record['demos'] = demos
                    false_reply.append(record)
                    save_json(
                        false_reply, f'./{self.save_root}/[{batch_type}_batch]{affx}_false_reply.json')
                    query_cnt += 1
                    continue

                break

            if query_cnt > max_cnt:
                continue

            curi = 0
            preds = []
            while True:
                if ans[curi] == 'y' and (ans[curi: curi+4] == 'yes.' or ans[curi: curi+4] == 'yes,'):
                    preds.append(1)
                if ans[curi] == 'n' and (ans[curi: curi+3] == 'no.' or ans[curi: curi+3] == 'no,'):
                    preds.append(0)
                curi += 1
                if curi >= len(ans) or len(preds) >= len(batch):
                    break

            record = {
                'ask': ask,
                'ans': ans,
                'batch_id': batch_id,
                'batch': batch,
                'ask_tok': tokens_num(self.encoding, ask),
                'ans_tok': tokens_num(self.encoding, ans),
                'preds': preds,
                'labs': [int(data.test_lab[i]) for i in batch]
            }
            if demo4batches is not None:
                record['demos'] = demos
            tol_records.append(record)
            save_json(
                tol_records, f'./{self.save_root}/[{batch_type}_batch]{affx}_tol_records.json')

        result = self._cal_results(tol_records)
        save_json(
            result, f'./{self.save_root}/[{batch_type}_batch]{affx}_result.json')
        if demo4batches is not None:
            unque_demo_cnt = len(
                set([d for demos in demo4batches for d in demos]))
            result['unique_demo_cnt'] = unque_demo_cnt
        save_json(
            result, f'./{self.save_root}/[{batch_type}_batch]{affx}_result.json')
        self.logger.info(
            f'============================= {batch_type} batch result =============================')
        self.logger.info(result)
        self.logger.info(
            f'=====================================================================================')
        return result

    def save_batches(self, batch_type='similar'):
        # load tol_records
        tol_records = open_json(
            f'./{self.save_root}/[{batch_type}_batch]_tol_records.json')
        batches = [record['batch'] for record in tol_records]
        save_json(
            batches, f'./{self.save_root}/[{batch_type}_batch]_batches.json')

    def load_batches(self, batch_type='similar'):
        batches = open_json(
            f'./{self.save_root}/[{batch_type}_batch]_batches.json')
        return batches

    def _allocate_1demo_for_1pair(self, batches, dis, topk=1):
        """return demos: List[List[int]], shape: (num_pairs, top_k)"""
        demos = []
        arg_dis = np.argsort(dis, axis=1)

        for b in batches:
            d = []
            for p in b:
                d = d + list(arg_dis[p][:topk])
                if sum([self.weight_d[int(i)] for i in d]) >= 4000-200-2000-20*len(b)-160:
                    break
            demos.append(list(d))
        return demos

    def _topk_demo_for_batches(self, batches, dis, topk):
        demos = []
        arg_dis = np.argsort(dis, axis=1)

        for b in batches:
            # b 为 pair 列表
            d_dis = {}
            for p in b:
                candidates = list(arg_dis[p][:topk])
                for did in candidates:
                    if did not in d_dis:
                        d_dis[did] = dis[p][did]
                    else:
                        d_dis[did] = min(d_dis[did], dis[p][did])
            tmp = sorted(d_dis.items(), key=lambda x: x[1], reverse=False)
            demos.append([int(tmp[i][0]) for i in range(topk)])
        return demos

    def _demo_cover_pair(self, pkgs, demo_dis_pecent, cover_cnt_all=1, cover_cnt_batch=1):
        demo_dis_thold = np.percentile(self.dis, demo_dis_pecent)
        p2ds, d2ps = self._get_p2ds_d2ps(self.dis, mx_dis=demo_dis_thold)
        all_pairs = [i for pkg in pkgs for i in pkg]
        all_demos = self._allocate_all_demos(d2ps, all_pairs=all_pairs,
                                             total_demos=list(
                                                 range(len(self.data.D_t1))),
                                             cover_cnt=cover_cnt_all)
        self.logger.info(f'tol_unique_demos: {len(all_demos)}')
        demos4pkgs = [
            self._allocate_demos_for_pkg(all_demo=all_demos, d2ps=d2ps,
                                         pkg=pkg, weight_d=self.weight_d,
                                         cover_cnt=cover_cnt_batch)
            for pkg in pkgs]
        self.logger.info(
            f'tol_demo_num: {sum([len(demos) for demos in demos4pkgs])}')
        return demos4pkgs

    def _allocate_demos_for_pkg(self, all_demo, d2ps, pkg, weight_d, cover_cnt=2, demo_token_limit=9999999999):
        pairs = {}
        demos = []
        for p in pkg:
            pairs[p] = cover_cnt

        while True:
            scores_of_demos = []
            for d in all_demo:
                score = 0
                cnt = 0
                for p in d2ps[d]:
                    if p in pairs:
                        cnt += 1
                if cnt != 0:
                    score = weight_d[d]/cnt
                    scores_of_demos.append((d, score))

            scores_of_demos = sorted(
                scores_of_demos, key=lambda x: x[1], reverse=False)

            selected_demo = -1
            for t in scores_of_demos:
                if t[0] not in demos:
                    selected_demo = t[0]
                    break

            if selected_demo != -1 and demo_token_limit >= weight_d[selected_demo]:
                demo_token_limit -= weight_d[selected_demo]
                demos.append(selected_demo)

                for p in d2ps[selected_demo]:
                    if p in pairs:
                        pairs[p] -= 1
                        if pairs[p] == 0:
                            pairs.pop(p)
            else:
                break

        return demos

    def _allocate_all_demos(self, d2ps, all_pairs, total_demos, cover_cnt=2):
        total_demos = copy.deepcopy(total_demos)
        pairs = {}
        demos = []
        for p in all_pairs:
            pairs[p] = cover_cnt

        while True:
            scores_of_demos = []
            for d in total_demos:
                cnt = 0
                for p in d2ps[d]:
                    if p in pairs:
                        cnt += 1
                if cnt != 0:
                    scores_of_demos.append((d, 1/cnt))

            scores_of_demos = sorted(
                scores_of_demos, key=lambda x: x[1], reverse=False)

            selected_demo = -1
            for t in scores_of_demos:
                if t[0] not in demos:
                    selected_demo = t[0]
                    break

            if selected_demo != -1:
                total_demos.remove(selected_demo)
                demos.append(selected_demo)

                for p in d2ps[selected_demo]:
                    if p in pairs:
                        pairs[p] -= 1
                        if pairs[p] == 0:
                            pairs.pop(p)
            else:
                break
        return demos

    def _get_p2ds_d2ps(self, dis, mx_dis, mi_dis=0):
        """
        dis: numpy.array, shape: (num_pairs, num_demo)
        dis_threshold: float
        """
        p2ds = {i: [] for i in range(dis.shape[0])}
        for i in range(dis.shape[0]):
            for j in range(dis.shape[1]):
                if dis[i][j] <= mx_dis and dis[i][j] >= mi_dis:
                    p2ds[i].append(j)

        d2ps = {i: [] for i in range(dis.shape[1])}
        for p in p2ds:
            for d in p2ds[p]:
                d2ps[d].append(p)

        return p2ds, d2ps

    def _get_p2ds(self, dis, mx_dis, mi_dis=0):
        """
        dis: numpy.array, shape: (num_pairs, num_demo)
        dis_threshold: float
        """
        p2ds = {i: [] for i in range(dis.shape[0])}
        for i in range(dis.shape[0]):
            for j in range(dis.shape[1]):
                if dis[i][j] <= mx_dis and dis[i][j] >= mi_dis:
                    p2ds[i].append(j)
            if len(p2ds[i]) <= 0:
                curind = np.argmin(dis[i])
                curdis = dis[i][curind]
                p2ds[i] = [curind]
        return p2ds

    def _cal_results(self, tol_records):
        tol_ask_toks = sum([record['ask_tok'] for record in tol_records])
        tol_ans_toks = sum([record['ans_tok'] for record in tol_records])
        preds = [pred for record in tol_records for pred in record['preds']]
        labs = [lab for record in tol_records for lab in record['labs']]
        p, n, tp, tn, fp, fn, recall, precision, f1 = cal_f1(preds, labs)
        result = {
            'tol_num': p+n,
            'tol_dollars': (tol_ask_toks/1000)*0.0015+(tol_ans_toks/1000)*0.002,
            'FN': fn,
            'TN': tn,
            'FP': fp,
            'TP': tp,
            'N': n,
            'P': p,
            'acc': (tp+tn)/(p+n+0.000001),
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'tol_ask_toks': tol_ask_toks,
            'tol_ans_toks': tol_ans_toks,
        }
        return result

    def _generate_2dicts(self, preds):
        pair2cluster, cluster2pair = {},  {}
        for pid, cluster in enumerate(preds):
            if cluster != -1:
                pair2cluster[pid] = cluster
                if cluster not in cluster2pair:
                    cluster2pair[cluster] = [pid]
                else:
                    cluster2pair[cluster].append(pid)

        max_cluster = max(cluster2pair.keys())
        # self.logger.info(f'tol clustered cluster num: {max_cluster+1}')
        # self.logger.info(f'tol data points included(clustered): {sum([len(cluster2pair[c]) for c in cluster2pair])}')

        for pid, cluster in enumerate(preds):
            if cluster == -1:
                max_cluster += 1
                pair2cluster[pid] = cluster
                cluster2pair[max_cluster] = [pid]

        # self.logger.info(f'including noisy sample cluster num: {max_cluster}')

        return pair2cluster, cluster2pair

    def _generate_tol_random_batch(self, batch_size):
        pids = list(range(len(self.data.test_lab)))
        shuffle(pids)
        batches = []
        # 将pids切成batchsize大小的batch
        for i in range(0, len(pids), batch_size):
            batch = pids[i:i+batch_size]
            batches.append(batch)
        self.logger.info(
            f'generate {len(batches)} random batches, tol pairs: {sum([len(batch) for batch in batches])}')
        return batches

    def _generate_tol_similar_batch(self, c2p, batch_size):
        remain_c2p = copy.deepcopy(c2p)  # !
        batches = []
        while True:
            if len(remain_c2p) <= 0:
                break

            remain_pairs = []
            for c in remain_c2p:
                remain_pairs += remain_c2p[c]

            if len(remain_pairs) <= batch_size:
                batches.append(remain_pairs)
                break

            batch = []
            while len(batch) < batch_size:
                for c in list(remain_c2p.keys()):
                    add_pairs = remain_c2p[c][:batch_size-len(batch)]
                    for p in add_pairs:
                        batch.append(p)
                        remain_c2p[c].remove(p)
                    if len(remain_c2p[c]) <= 0:
                        remain_c2p.pop(c)
            batches.append(batch)

        self.logger.info(
            f'generate {len(batches)} similar batches, tol pairs: {sum([len(batch) for batch in batches])}')
        return batches

    def _generate_tol_diverse_batch(self, c2p, batch_size):
        remain_c2p = copy.deepcopy(c2p)
        batches = []
        while True:
            if len(remain_c2p) <= 0:
                break
            if len(remain_c2p) < batch_size:
                remain_pairs = []
                for c in remain_c2p:
                    remain_pairs += remain_c2p[c]
                if len(remain_pairs) <= batch_size:
                    batches.append(remain_pairs)
                    break

                batch = []

                while len(batch) < batch_size:
                    for c in list(remain_c2p.keys()):
                        t = remain_c2p[c][0]
                        batch.append(t)
                        remain_c2p[c].remove(t)
                        remain_pairs.remove(t)
                        if len(remain_c2p[c]) <= 0:
                            remain_c2p.pop(c)
                        if len(batch) >= batch_size:
                            break

                batches.append(batch)
                continue
            c_lis = sample(list(remain_c2p.keys()), batch_size)
            batch = []
            for c in c_lis:
                t = remain_c2p[c][0]
                batch.append(t)
                remain_c2p[c].remove(t)
                if len(remain_c2p[c]) <= 0:
                    remain_c2p.pop(c)
            batches.append(batch)
        self.logger.info(
            f'generate {len(batches)} diverse batches, tol pairs: {sum([len(batch) for batch in batches])}')

        return batches

    def _dis_of_pairdemos_Eucli(self, pair_arr, demo_arr):
        """pair_arr: (num_pairs, num_feas), demo_arr: (num_demo, num_feas)"""
        dis = np.zeros((pair_arr.shape[0], demo_arr.shape[0]))
        for i in range(pair_arr.shape[0]):
            dis[i] = self._dis_of_vecs_Eucli(
                np.tile(pair_arr[i], (demo_arr.shape[0], 1)), demo_arr)
        return dis

    def _dis_of_vecs_Eucli(self, arr1, arr2):
        dis = np.linalg.norm(arr1-arr2, axis=1)
        return dis

    def _pca(self, arrs):
        return self.pca.fit_transform(arrs)

    def _extact_fea_arrs(self, t1s, t2s, dn, encoder=None, sim_type='StructureAware', sim='ratio'):
        if sim_type == 'StructureAware':
            feas = get_fea_of_entries(
                t1s, t2s, dn, sim_mode='PROP_BASED', encoder=encoder, sim_func=sim)
        else:
            feas = get_fea_of_entries(
                t1s, t2s, dn, sim_mode='SEMANTIC_BASED', encoder=encoder)
        return np.array([[fea[k] for k in fea] for fea in feas]) if sim_type == 'StructureAware' else feas
