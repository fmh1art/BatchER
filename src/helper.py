import tiktoken
import Levenshtein
import json
import logging
import os
import re
import json
import torch


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.handlers.clear()
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8', mode='a')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


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
    with open(path, "r", encoding='ISO-8859-1') as f:
        data = json.load(f)
    return data


def save_json(a, fn):
    b = json.dumps(a)
    f2 = open(fn, 'w')
    f2.write(b)
    f2.close()


def choose_device(device_num=-1):
    return torch.device(f"cuda:{device_num}" if device_num >= 0 else "cpu")


def cal_tokens(s):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    codelist = encoding.encode(s)
    return len(codelist)


def tokens_num(encoding, s):
    return len(encoding.encode(s))


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


def get_fea_of_entries(t1, t2, dn, sim_mode='PROP_BASED', sim_func='ratio', encoder=None, bm25=None):
    if sim_mode == 'PROP_BASED':
        if encoder != None:
            return get_fea_based_on_properties_with_encoder(t1, t2, dn, sim_func=sim_func, encoder=encoder)
        else:
            return get_fea_based_on_properties(t1, t2, dn, sim_func=sim_func)
    elif sim_mode == 'SEMANTIC_BASED':
        return get_fea_based_on_semantic(t1, t2, dn, encoder)


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



def get_fea_based_on_properties_with_encoder(t1, t2, dn, sim_func='ratio', encoder=None):
    """
    sim_func: jaro_winkler, ratio, BM25
    """
    keys = None
    for a, b in zip(t1, t2):
        ad = json.loads(add_quotation_mark_to_key(dn, '{'+a+'}'))
        bd = json.loads(add_quotation_mark_to_key(dn, '{'+b+'}'))
        if keys == None:
            keys = list(ad.keys())
        elif len(keys) < len(ad.keys()):
            keys = list(ad.keys())
        elif len(keys) < len(bd.keys()):
            keys = list(bd.keys())

    fea = [{k: 0 for k in keys} for _ in range(len(t1))]

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


def semantic_based_sims(feas1, feas2):
    """
    fea1: numpy.array, shape=(n, num_hidden)
    fea1: numpy.array, shape=(n, num_hidden)
    """
    embeddings1 = torch.tensor(feas1)
    embeddings2 = torch.tensor(feas2)
    similarity = torch.cosine_similarity(embeddings1, embeddings2, dim=1)
    return similarity.tolist()


def get_fea_based_on_properties(t1, t2, dn, sim_func='ratio'):
    """
    sim_func: jaro_winkler, ratio, BM25
    """
    keys = None
    for a, b in zip(t1, t2):
        ad = json.loads(add_quotation_mark_to_key(dn, '{'+a+'}'))
        bd = json.loads(add_quotation_mark_to_key(dn, '{'+b+'}'))
        if keys == None:
            keys = list(ad.keys())
        elif len(keys) < len(ad.keys()):
            keys = list(ad.keys())
        elif len(keys) < len(bd.keys()):
            keys = list(bd.keys())

    fea = [{k: 0 for k in keys} for _ in range(len(t1))]

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
                    if sim_func == 'ratio':
                        sco = Levenshtein.ratio(av, bv)
                    elif sim_func == 'jaro_winkler':
                        sco = Levenshtein.jaro_winkler(av, bv)
                    fea[i][k] = sco

    return fea


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
        if keys == None:
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
        s_pair = serialize_entity(
            ad, keys)+' [SEP] '+serialize_entity(bd, keys)
        serialized_pairs.append(s_pair)
    return encoder.encode(serialized_pairs)


def remov_novel_keys(dn, a, b):
    ad = json.loads(add_quotation_mark_to_key(dn, '{'+a+'}'))
    bd = json.loads(add_quotation_mark_to_key(dn, '{'+b+'}'))
    for k in list(ad.keys()):
        if len(ad[k]) == 0 or len(bd[k]) == 0 or '(missing)' in ad[k] or '(missing)' in bd[k]:
            ad.pop(k)
            bd.pop(k)
    return json.dumps(ad).replace('{', '').replace('}', ''), json.dumps(bd).replace('{', '').replace('}', '')


def filtering(dataset_name, a):
    ad = json.loads(add_quotation_mark_to_key(dataset_name, '{'+a+'}'))
    remaining_keys = {
        'em-beer': ['Beer_Name', 'Brew_Factory_Name'],
        'em-ia': ['Song_Name'],
        'em-fz': ['name', 'addr', 'type', 'class'],
        'em-wa': ['modelno', 'title'],
        'em-ds': ['title', 'authors', 'venue', 'year'],
        'abt-buy': ['name'],
        'em-da-dirty': ['title', 'authors', 'venue', 'year'],
        'em-da': ['title', 'authors', 'venue', 'year'],
        'em-ag': ['title', 'manufacturer']
    }

    res_d = {}

    for k in remaining_keys[dataset_name]:
        if k in ad:
            res_d[k] = ad[k]

    return json.dumps(res_d).replace('{', '').replace('}', '')


def filter_str(desstr):
    res = re.sub("[+\-\!\/$%^*+\']+|[+——?【】“”！，。？、~@#￥%……&*（）`]+", "", desstr)
    return res.replace('  ', '').replace('  ', ' ').replace('[', '(').replace(']', ')').replace("' ", "").replace(" '", "")
