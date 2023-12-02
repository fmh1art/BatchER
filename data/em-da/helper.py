import random
import tiktoken
import Levenshtein
import copy
import json
import numpy as np
from pyLSHash import LSHash
import logging
import os, re, sys
import openai
import time
import pickle as pkl
import json
import pandas as pd
from random import sample, shuffle
import torch
import torch.nn.functional as F

def logger_config(log_path,logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    logger.handlers.clear()
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8',mode='a')
    handler.setLevel(logging.DEBUG)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def open_json(path):
    with open(path,"r",encoding='ISO-8859-1') as f:
        data = json.load(f)
    return data

def add_quotation_mark_to_key(dataset_name, s):
    s = s.replace('\\', '')
    if dataset_name == 'em-ia' or dataset_name == 'em-ia-dirty':
        return s.replace('Song_Name:', '"Song_Name":')\
            .replace('Artist_Name:', '"Artist_Name":')\
                .replace('Album_Name:', '"Album_Name":')\
                    .replace('Genre:', '"Genre":').replace('Price:', '"Price":').\
                        replace('CopyRight:', '"CopyRight":').replace('Time:', '"Time":').\
                            replace('Released:', '"Released":')
    elif dataset_name == 'em-beer' or dataset_name == 'em-beer-dirty':
        return s.replace('Beer_Name:', '"Beer_Name":')\
            .replace('Brew_Factory_Name:', '"Brew_Factory_Name":')\
                .replace('Style:', '"Style":')\
                    .replace('ABV:', '"ABV":')
    elif dataset_name == 'em-fz' or dataset_name == 'em-fz-dirty':
        return s.replace('name:', '"name":')\
            .replace('addr:', '"addr":')\
                .replace('city:', '"city":')\
                .replace('type:', '"type":')\
                .replace('class:', '"class":')\
                    .replace('phone:', '"phone":')
    elif dataset_name == 'em-wa' or dataset_name == 'em-wa-dirty':
        return s.replace('title:', '"title":')\
            .replace('category:', '"category":')\
                .replace('brand:', '"brand":')\
                .replace('modelno:', '"modelno":')\
                .replace('price:', '"price":')
    elif dataset_name == 'em-ds' or dataset_name == 'em-da-dirty' or dataset_name == 'em-da':
        return s.replace('title:', '"title":')\
            .replace('authors:', '"authors":')\
                .replace('venue:', '"venue":')\
                .replace('year:', '"year":')
    elif dataset_name == 'abt-buy' or dataset_name == 'abt-buy-dirty':
        return s.replace('name:', '"name":')\
            .replace('description:', '"description":')\
                .replace('price:', '"price":')
    elif dataset_name == 'em-ag' or dataset_name == 'em-ag-dirty':
        return s.replace('title:', '"title":')\
            .replace('manufacturer:', '"manufacturer":')\
                .replace('price:', '"price":')\

def get_fea_of_entries(t1, t2, dn, sim_mode='PROP_BASED', sim_func='ratio', encoder=None, bm25=None):
    if sim_mode == 'PROP_BASED':
        if encoder != None:
            return get_fea_based_on_properties_with_encoder(t1, t2, dn, sim_func=sim_func, encoder=encoder)
        else:
            return get_fea_based_on_properties(t1, t2, dn, sim_func=sim_func)
    elif sim_mode == 'SEMANTIC_BASED': # 都输出numpy.array
        return get_fea_based_on_semantic(t1, t2, dn, encoder)

def get_fea_based_on_properties_with_encoder(t1, t2, dn, sim_func='ratio', encoder=None):
    """
    sim_func: jaro_winkler, ratio, BM25
    """
    keys = None
    for a, b in zip(t1, t2):
        ad = json.loads(add_quotation_mark_to_key(dn, '{'+a+'}'))
        bd = json.loads(add_quotation_mark_to_key(dn, '{'+b+'}'))
        if keys==None:
            keys = list(ad.keys())
        elif len(keys) < len(ad.keys()):
            keys = list(ad.keys())
        elif len(keys) < len(bd.keys()):
            keys = list(bd.keys())

    fea = [{k:0 for k in keys} for _ in range(len(t1))]
    
    data = list(zip(t1, t2))
    item1_lis = []
    item2_lis = []
    for i, t in enumerate(data):
        a, b = t[0], t[1]
        ad = json.loads(add_quotation_mark_to_key(dn, '{'+a+'}'))
        bd = json.loads(add_quotation_mark_to_key(dn, '{'+b+'}'))
        for j, k in enumerate(keys):
            if k in ad and k in bd:
                av = ad[k]
                bv = bd[k]
                if (len(av) != 0 and len(bv) != 0) or ('(missing)' not in av and '(missing)' not in bv):
                    item1_lis.append(av)
                    item2_lis.append(bv)

    em1 = encoder.encode(item1_lis)
    em2 = encoder.encode(item2_lis)
    sims = semantic_based_sims(em1, em2)

    cur_ind = 0
    for i, t in enumerate(data):
        a, b = t[0], t[1]
        ad = json.loads(add_quotation_mark_to_key(dn, '{'+a+'}'))
        bd = json.loads(add_quotation_mark_to_key(dn, '{'+b+'}'))
        for j, k in enumerate(keys):
            if k in ad and k in bd:
                fea[i][k] = 0
                av = ad[k]
                bv = bd[k]
                if len(av) == 0 or '(missing)' in av:
                    fea[i][k] -= 1
                if len(bv) == 0 or '(missing)' in bv:
                    fea[i][k] -= 1
                if (len(av) != 0 and len(bv) != 0) or ('(missing)' not in av and '(missing)' not in bv):
                    fea[i][k] = sims[cur_ind]
                    cur_ind += 1
    
    return fea

def get_fea_based_on_properties(t1, t2, dn, sim_func='ratio'):
    """
    sim_func: jaro_winkler, ratio, BM25
    """
    keys = None
    for a, b in zip(t1, t2):
        ad = json.loads(add_quotation_mark_to_key(dn, '{'+a+'}'))
        bd = json.loads(add_quotation_mark_to_key(dn, '{'+b+'}'))
        if keys==None:
            keys = list(ad.keys())
        elif len(keys) < len(ad.keys()):
            keys = list(ad.keys())
        elif len(keys) < len(bd.keys()):
            keys = list(bd.keys())

    fea = [{k:0 for k in keys} for _ in range(len(t1))]
    
    data = list(zip(t1, t2))
    for i, t in enumerate(data):
        a, b = t[0], t[1]
        ad = json.loads(add_quotation_mark_to_key(dn, '{'+a+'}'))
        bd = json.loads(add_quotation_mark_to_key(dn, '{'+b+'}'))
        for j, k in enumerate(keys):
            if k in ad and k in bd:
                fea[i][k] = 0
                av = ad[k]
                bv = bd[k]

                emp1 = (len(av) == 0 or '(missing)' in av)
                emp2 = (len(bv) == 0 or '(missing)' in bv)
                if emp1 and emp2:
                    fea[i][k] = 1
                    continue
                if emp1 or emp2:
                    fea[i][k] = 0
                    continue
                if not emp1 and not emp2:
                    if sim_func=='ratio':
                        sco = Levenshtein.ratio(av, bv)
                    elif sim_func=='jaro_winkler':
                        sco = Levenshtein.jaro_winkler(av, bv)
                    fea[i][k] = sco
                
    return fea

main_key = {
                'em-beer': 'Beer_Name',
                'em-ia': 'Song_Name',
                'em-fz': 'name',
                'em-wa': 'title',
                'em-ds': 'title',
                'abt-buy': 'name',
                'em-da-dirty': 'title',
                'em-da': 'title',
                'em-ag': 'title'
}

def get_fea_based_on_semantic(t1, t2, dn, encoder):
    
    def serialize_entity(e, keys):
        s = ''
        for k in keys:
            if k in e:
                s += ' [COL] '+k+' [VAL] '+str(e[k])
            else:
                s += ' [COL] '+k
        return s

    keys = None
    for a, b in zip(t1, t2):
        ad = json.loads(add_quotation_mark_to_key(dn, '{'+a+'}'))
        bd = json.loads(add_quotation_mark_to_key(dn, '{'+b+'}'))
        if keys==None:
            keys = list(ad.keys())
        elif len(keys) < len(ad.keys()):
            keys = list(ad.keys())
        elif len(keys) < len(bd.keys()):
            keys = list(bd.keys())

    data = list(zip(t1, t2))
    serialized_pairs = []
    for i, t in enumerate(data):
        a, b = t[0], t[1]
        ad = json.loads(add_quotation_mark_to_key(dn, '{'+a+'}'))
        bd = json.loads(add_quotation_mark_to_key(dn, '{'+b+'}'))
        s_pair = serialize_entity(ad, keys)+' [SEP] '+serialize_entity(bd, keys)
        serialized_pairs.append(s_pair)
    return encoder.encode(serialized_pairs)

def dist_func(lis1, lis2):
    sims = [0]
    for i, (x1, x2) in enumerate(zip(lis1, lis2)):
        if x1 > 0 and x2 > 0:
            sims.append( 1-abs(x1-x2) )
        else: # 如果有一个pair的vector中的值为0
            if x1==x2:
                sims.append(1)
            else:
                sims.append(0)
    sim = min(sims)*0.1+(sum(sims)/len(sims))*0.9
    sim = max(sim, 0)
    return 1-sim

def sim_func(fea1, fea2, sim_mode='PROP_BASED'):
    if sim_mode == 'PROP_BASED':
        return prop_based_sim(fea1, fea2)
    elif sim_mode == 'SEMANTIC_BASED':
        return semantic_based_sim(fea1, fea2)

def semantic_based_sim(fea1, fea2):
    """
    fea1: numpy.array, shape=(num_hidden)
    fea1: numpy.array, shape=(num_hidden)
    """
    embeddings1 = torch.tensor(fea1).reshape(shape=(1, fea1.shape[-1]))
    embeddings2 = torch.tensor(fea2).reshape(shape=(1, fea1.shape[-1]))
    similarity = torch.cosine_similarity(embeddings1, embeddings2, dim=1)
    return similarity.tolist()[0]

def sims_of_feas(feas1, feas2, sim_mode='PROP_BASED'):
    if sim_mode == 'PROP_BASED':
        return [prop_based_sim(fea1, fea2) for fea1, fea2 in zip(feas1, feas2)]
    elif sim_mode == 'SEMANTIC_BASED':
        return semantic_based_sims(feas1, feas2)

def semantic_based_sims(feas1, feas2):
    """
    fea1: numpy.array, shape=(n, num_hidden)
    fea1: numpy.array, shape=(n, num_hidden)
    """
    embeddings1 = torch.tensor(feas1)
    embeddings2 = torch.tensor(feas2)
    similarity = torch.cosine_similarity(embeddings1, embeddings2, dim=1)
    return similarity.tolist()

def prop_based_sim(fea1, fea2):
    sims = []
    if 'modelno' in fea1 and (fea1['modelno'] >= 0 or fea2['modelno'] >= 0):
        x1 = fea1['modelno']
        x2 = fea2['modelno']
        # if min(x1, x2) < 0:
        #     return 0
        if x1 + x2 >= 2:
            return 1
        # if x1 >= 1 and x2 <= 0.8:
        #     return 0
        
    for k in fea1:
        s = 0
        x1 = fea1[k]
        x2 = fea2[k]
        
        if x1 >= 0 and x2 >= 0:
            s = 1-abs(x1-x2)

            # if k=='Style':
            #     s *= 0.8
            # if k=='ABV':
            #     s *= 0.8

            sims.append(s)
        else: 
            # if x1 != x2:    
            #     sims.append(0)
            continue

    # sim = min(sims)*0.5+(sum(sims)/len(sims))*0.5
    # sim = min(sims)*0.4+(sum(sims)/len(sims))*0.6
    # sim = min(sims)*0.6+(sum(sims)/len(sims))*0.4
    sim = min(sims)*0.1+(sum(sims)/len(sims))*0.9
    # sim = min(sims)
    # sim = sum(sims)/len(sims)
    
    sim = max(sim, 0)
    return sim

def getCC(g, sim):
    s = 0
    for c in g:
        for i in range(len(c)):
            for j in range(i+1, len(c)):
                p1 = c[i]
                p2 = c[j]
                s += sim[p1][p2]
    
    for i in range(len(g)):
        for j in range(i+1, len(g)):
            c1 = g[i]
            c2 = g[j]
            for p1 in c1:
                for p2 in c2:
                    s += 1-sim[p1][p2]
    
    return s

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		# print("---  new folder...  ---")
		# print("---  OK  ---")
	# else:
		# print("---  folder exists!  ---")

def is_float(s):
    isFloat = False
    try:
        float(s)
    except:
        isFloat = False
    else:
        isFloat = True
    return isFloat

def open_json(path):
    with open(path,"r",encoding='ISO-8859-1') as f:
        data = json.load(f)
    return data

def save_json(a, fn):
    b = json.dumps(a)
    f2 = open(fn, 'w')
    f2.write(b)
    f2.close() 

def choose_device(device_num=-1):
    return torch.device(f"cuda:{device_num}" if device_num>=0 else "cpu")

def get_file_name(dn=None, sim_rat=None, demo_sim_rat=None, cover_cnt=None, temp_cnt=None, 
                  max_cluster=None, reverse_group=None, add_extra_demo=None, sim_minus=None, 
                  check_exp=False, group_mode=None, demo_mode=None, pair_per_group=None, demo_cnt=None,
                  shuffle_pairs = None, MAX_TOK=None, max_pair_per_group=None, property_descrip=None):
    its = []
    if check_exp:
        its.append(f'【check_exp】')
        if group_mode is not None:
            its.append(f'gm.{group_mode}')
        if demo_mode is not None:
            its.append(f'dm.{demo_mode}')
        if pair_per_group is not None:
            its.append(f'ppg.{pair_per_group}')
        if demo_cnt is not None:
            its.append(f'dc.{demo_cnt}')
    if dn is not None:
        its.append(f'dn.{dn}')
    if sim_rat is not None:
        its.append(f'sr.{sim_rat}')
    if demo_sim_rat is not None:
        its.append(f'dsr.{demo_sim_rat}')
    if cover_cnt is not None:
        its.append(f'cc.{cover_cnt}')
    if temp_cnt is not None:
        its.append(f'tc.{temp_cnt}')
    if max_cluster is not None:
        its.append(f'mc.{max_cluster}')
    if reverse_group is not None:
        its.append(f'rg.{reverse_group}')
    if add_extra_demo is not None:
        its.append(f'aed.{add_extra_demo}')
    if sim_minus is not None:
        its.append(f'sm.{sim_minus}')
    if shuffle_pairs is not None:
        its.append(f'sp.{shuffle_pairs}')
    if MAX_TOK is not None:
        its.append(f'MT.{MAX_TOK}')
    if max_pair_per_group is not None:
        its.append(f'mppg.{max_pair_per_group}')
    if property_descrip is not None:
        its.append(f'pd.{property_descrip}')

    return '_'.join(its)

def allocate_chatER_alldemo(demo2pair, all_pairs, weight_d, total_demos, cover_cnt=2, demo_token_limit=9999999999):
    pairs = {}
    demos = []
    for p in all_pairs:
        pairs[p] = cover_cnt

    while True:
        scores_of_demos = []
        for d in total_demos:
            score = 0
            cnt = 0
            for p in demo2pair[d]:
                if p in pairs:
                    cnt += 1
            if cnt != 0:
                score = weight_d[d]/cnt
                scores_of_demos.append((d, score))

        scores_of_demos = sorted(scores_of_demos, key=lambda x:x[1], reverse=False)

        selected_demo = -1
        for t in scores_of_demos:
            if t[0] not in demos:
                selected_demo = t[0]
                break

        if selected_demo != -1 and demo_token_limit >= weight_d[selected_demo]:
            demo_token_limit -= weight_d[selected_demo]
            demos.append(selected_demo)
            
            for p in demo2pair[selected_demo]:
                if p in pairs:
                    pairs[p] -= 1
                    if pairs[p] == 0:
                        pairs.pop(p)
        else:
            break


    while len(pairs) != 0:
        scores_of_demos = []
        for d in demo2pair:
            score = 0
            cnt = 0
            for p in demo2pair[d]:
                if p in pairs:
                    cnt += 1
            if cnt != 0:
                score = weight_d[d]/cnt
                scores_of_demos.append((d, score))

        scores_of_demos = sorted(scores_of_demos, key=lambda x:x[1], reverse=False)

        selected_demo = -1
        for t in scores_of_demos:
            if t[0] not in demos:
                selected_demo = t[0]
                break

        if selected_demo != -1 and demo_token_limit >= weight_d[selected_demo]:
            demo_token_limit -= weight_d[selected_demo]
            demos.append(selected_demo)
            
            for p in demo2pair[selected_demo]:
                if p in pairs:
                    pairs[p] -= 1
                    if pairs[p] == 0:
                        pairs.pop(p)
        else:
            break
    
    return demos

def allocate_demo(all_demo, demo2pair, cluster, weight_d, cover_cnt=2, demo_token_limit=9999999999):
    pairs = {}
    demos = []
    for p in cluster:
        pairs[p] = cover_cnt

    while True:
        scores_of_demos = []
        for d in all_demo:
            score = 0
            cnt = 0
            for p in demo2pair[d]:
                if p in pairs:
                    cnt += 1
            if cnt != 0:
                score = weight_d[d]/cnt
                scores_of_demos.append((d, score))

        scores_of_demos = sorted(scores_of_demos, key=lambda x:x[1], reverse=False)

        selected_demo = -1
        for t in scores_of_demos:
            if t[0] not in demos:
                selected_demo = t[0]
                break

        if selected_demo != -1 and demo_token_limit >= weight_d[selected_demo]:
            demo_token_limit -= weight_d[selected_demo]
            demos.append(selected_demo)
            
            for p in demo2pair[selected_demo]:
                if p in pairs:
                    pairs[p] -= 1
                    if pairs[p] == 0:
                        pairs.pop(p)
        else:
            break

    return demos


# def allocate_demo(all_demos, demo2pair, cluster, weight_d, total_demos, cover_cnt=2, demo_token_limit=9999999999):
#     pairs = {}
#     demos = []
#     for p in cluster:
#         pairs[p] = cover_cnt

#     while True:
#         scores_of_demos = []
#         for d in total_demos:
#             score = 0
#             cnt = 0
#             for p in demo2pair[d]:
#                 if p in pairs:
#                     cnt += 1
#             if cnt != 0:
#                 score = weight_d[d]/cnt
#                 scores_of_demos.append((d, score))

#         scores_of_demos = sorted(scores_of_demos, key=lambda x:x[1], reverse=False)

#         selected_demo = -1
#         for t in scores_of_demos:
#             if t[0] not in demos:
#                 selected_demo = t[0]
#                 break

#         if selected_demo != -1 and demo_token_limit >= weight_d[selected_demo]:
#             demo_token_limit -= weight_d[selected_demo]
#             demos.append(selected_demo)
            
#             for p in demo2pair[selected_demo]:
#                 if p in pairs:
#                     pairs[p] -= 1
#                     if pairs[p] == 0:
#                         pairs.pop(p)
#         else:
#             break


#     while len(pairs) != 0:
#         scores_of_demos = []
#         for d in demo2pair:
#             score = 0
#             cnt = 0
#             for p in demo2pair[d]:
#                 if p in pairs:
#                     cnt += 1
#             if cnt != 0:
#                 score = weight_d[d]/cnt
#                 scores_of_demos.append((d, score))

#         scores_of_demos = sorted(scores_of_demos, key=lambda x:x[1], reverse=False)

#         selected_demo = -1
#         for t in scores_of_demos:
#             if t[0] not in demos:
#                 selected_demo = t[0]
#                 break

#         if selected_demo != -1 and demo_token_limit >= weight_d[selected_demo]:
#             demo_token_limit -= weight_d[selected_demo]
#             demos.append(selected_demo)
            
#             for p in demo2pair[selected_demo]:
#                 if p in pairs:
#                     pairs[p] -= 1
#                     if pairs[p] == 0:
#                         pairs.pop(p)
#         else:
#             break
    
#     return demos

def allocate_random_predemo(demo_inds, demo_cnt):
    return sample(demo_inds, 
                  min(len(demo_inds), demo_cnt)
                  )

def allocate_diverse_predmo(fea_demos, demo_cnt):
    demos = []

    root = sample([i for i in range(len(fea_demos))], 1)[0]
    demos.append(root)
    for _ in range(1, demo_cnt):
        sim_lis = []
        for i in range(len(fea_demos)):
            if i in demos:
                continue
            sim_lis.append(
                (i, sum([sim_func(fea_demos[i], fea_demos[j]) for j in demos]))
                )
        sim_lis = sorted(sim_lis, key=lambda x:x[1], reverse=False)
        demos.append(sim_lis[0][0])

    return demos

def allocate_relevant_demo(pair2demo, cluster, cover_cnt):
    demos = []
    for p in cluster:
        demos += pair2demo[p][:cover_cnt]
    return demos

def cal_tokens(s):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    codelist = encoding.encode(s)
    return len(codelist)

def tokens_num(encoding, s):
    return len(encoding.encode(s))

def generate_groups(order, cnt_pair, MAX_TOK, weight, sim, sim_thold, 
                    max_cluster=9999, temp=-1, sim_minux=0.2, max_pair_per_cluster=9999,
                    max_pair_per_group = 99999):
    
    groups = []
    while True:
 
        if len(order) == 0:
            break
        
        g = []
        rep_c = []

        room = MAX_TOK

        ind = 0
        # generate one group
        while len(order) != 0:
            if ind >= len(order) or room-150*len(g) < weight[order[ind]]:
                break
            
            cur = order[ind] # try to add cur to g
            
            mi = 1
            ma = 0
            ma_ind = -1
            for i, c in enumerate(rep_c):
                if mi > sim[cur][c]:
                    mi = sim[cur][c]
                if ma < sim[cur][c]:
                    ma = sim[cur][c]
                    ma_ind = i
                
            add = False
            if ma_ind != -1 and temp != -1:
                # pair_tol_cnt = sum([len(c) for c in g])
                pair_tol_cnt = len(g[ma_ind])
                cur_thold = sim_thold + (pair_tol_cnt/temp)
            else:
                cur_thold = sim_thold
            if (ma > cur_thold and len(g[ma_ind]) < max_pair_per_cluster) or (ma <= cur_thold-sim_minux and len(g) < max_cluster):
                add = True
            if sum([len(c) for c in g]) >= max_pair_per_group:
                add = False
            if not add:
                ind += 1
                continue
            if ma <= cur_thold-sim_minux and len(g) < max_cluster:
                g.append([cur])
                rep_c.append(cur)

                room -= weight[cur]
                order.remove(cur)
            elif ma > cur_thold and len(g[ma_ind]) < max_pair_per_cluster:
                g[ma_ind].append(cur)

                room -= weight[cur]
                order.remove(cur)

        # print(g,'-====', order)
        groups.append(g)

    return groups

def generate_group_for_ckexp(order, sim, sim_thold, mode='random', pair_per_group=3):
    """
    mode: random, similarity, diversity
    """

    if mode == 'random':
        pair_inds = [i for i in range(len(order))]
        shuffle(pair_inds)
        return generate_random_groups(pair_inds, pair_per_group)
    elif mode == 'similarity':
        return generate_similarity_groups(order, sim, sim_thold, pair_per_group)
    elif mode == 'diversity':
        return generate_diversity_groups(order, sim, sim_thold, pair_per_group)

def generate_random_groups(order, pair_per_group=3):
    groups = []
    for i in range(0, len(order), pair_per_group):
        groups.append([order[i:i+pair_per_group]])
    
    return groups

def generate_similarity_groups(order, sim, sim_thold, pair_per_group=3):
    groups = []

    while True:
        if len(order) == 0:
            break

        c = [order[0]]
        root = c[0]
        for t in order[1:]:
            if sim[root][t] >= sim_thold and len(c) < pair_per_group:
                c.append(t)
            if len(c) >= pair_per_group:
                break
        
        for t in c:
            order.remove(t)

        groups.append([c])

    return groups

def generate_diversity_groups(order, sim, sim_thold, pair_per_group=3):
    
    groups = []

    while True:
        if len(order) == 0:
            break

        c = [order[0]]
        for t in order[1:]:
            if len(c) >= pair_per_group:
                break
            add = True
            for p in c:
                if sim[p][t] >= sim_thold:
                    add = False
                    break
            if add:
                c.append(t)
        
        for t in c:
            order.remove(t)

        groups.append([c])

    return groups

def cal_f1(preds, labs):
    p, n, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0
    for pred, lab in zip(preds, labs):
        if lab == 1:
            p += 1
            if pred == 1:
                tp += 1
            elif pred == 0:
                fn += 1
        else:
            n += 1
            if pred == 0:
                tn += 1
            elif pred == 1:
                fp += 1

    div_safe = 0.000001
    recall = tp/(p+div_safe)
    
    precision = tp/(tp+fp+div_safe)
    f1 = 2*recall*precision/(recall + precision + div_safe)
    return p, n, tp, tn, fp, fn, recall, precision, f1


def dis_vec(v1, v2):
    lis = [abs(e1-e2) for e1, e2 in zip(v1, v2)]
    return sum(lis)/len(lis)

def compute_avgdis(feas):
    keys = []
    for fea in feas:
        for k in fea:
            if k not in keys:
                keys.append(k)
                
    vecs = []
    for fea in feas:
        vec = []
        for k in keys:
            if k in fea:
                vec.append(fea[k])
            else:
                vec.append(-1)
                print('notin')
        vecs.append(vec)
    
    avgs = np.average(np.array(vecs), axis=0).tolist()
    dis_lis = [dis_vec(avgs, v) for v in vecs]
    return sum(dis_lis)/len(dis_lis)

Item_type = {
    'em-beer': 'Beer',
    'em-ia': 'Song',
    'em-fz': 'Restraunt',
    'em-wa': 'Product',
    'em-ds': 'Paper',
    'abt-buy': 'Product',
    'em-da-dirty': 'Paper',
    'em-da': 'Paper',
    'em-ag': 'Product'
}


def remov_novel_keys(dn, a, b):
    ad = json.loads(add_quotation_mark_to_key(dn, '{'+a+'}'))
    bd = json.loads(add_quotation_mark_to_key(dn, '{'+b+'}'))
    for k in list(ad.keys()):
        if len(ad[k]) == 0 or len(bd[k]) == 0 or '(missing)' in ad[k] or '(missing)' in bd[k]:
            ad.pop(k)
            bd.pop(k)
    return json.dumps(ad).replace('{', '').replace('}', ''), json.dumps(bd).replace('{', '').replace('}', '')