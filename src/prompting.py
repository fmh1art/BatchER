import os
import openai
import json
import copy
import random
import time
from random import sample, shuffle
from src.helper import *

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


class Prompt:

    Item_type = ""

    @staticmethod
    def generate_prompt_for_zeroshot_package(task_description, pairs, remove_noval_keys=False, dataset_name='em-ag'):

        pair_txt = []
        pcnt = 0
        for p in pairs:
            p = list(p)
            a = p[0].replace('(missing)', '')
            b = p[1].replace('(missing)', '')
            if remove_noval_keys:
                a, b = remov_novel_keys(dataset_name, a, b)
            pcnt += 1
            pair_txt.append(
                f'Question {pcnt}:\n {Prompt.Item_type} A is {a}\n {Prompt.Item_type} B is {b}')

        sent = 'question above' if pcnt == 1 else f'above {pcnt} questions'
        pair_txt.append(f'\nUse domain knowledge of {Prompt.Item_type}s to help understand the text and answer the {sent} in the format: For Question i, Yes, {Prompt.Item_type} A and {Prompt.Item_type} B are the same {Prompt.Item_type.lower()}./No, {Prompt.Item_type} A and {Prompt.Item_type} B are different {Prompt.Item_type.lower()}s. For Question i+1, (repeat the above procedures)')
        pair_prompt = '\n'.join(pair_txt)

        prompt = '\n\n'.join([task_description, pair_prompt])
        return prompt

    @staticmethod
    def generate_prompt_for_gourp_pre_exp(task_description, demos, pairs):
        demo_txt = []

        for i, d in enumerate(demos):
            d = list(d)
            d[0] = d[0].replace('(missing)', '')
            d[1] = d[1].replace('(missing)', '')
            demo_txt.append(
                f'Demonstration {i+1}:\n{Prompt.Item_type} A is {d[0]}\n{Prompt.Item_type} B is {d[1]}')
            if d[2] == '1':
                demo_txt.append(
                    f'Yes, {Prompt.Item_type} A and {Prompt.Item_type} B are the same {Prompt.Item_type.lower()}.')
            else:
                demo_txt.append(
                    f'No, {Prompt.Item_type} A and {Prompt.Item_type} B are different {Prompt.Item_type.lower()}s.')

        demo_prompt = '\n'.join(demo_txt)

        pair_txt = []

        for i, p in enumerate(pairs):
            p = list(p)
            p[0] = p[0].replace('(missing)', '')
            p[1] = p[1].replace('(missing)', '')
            pair_txt.append(
                f'Question {i+1}:\n {Prompt.Item_type} A is {p[0]}\n {Prompt.Item_type} B is {p[1]}')

        pcnt = len(pairs)
        sent = 'question above' if pcnt == 1 else f'above {pcnt} questions'
        pair_txt.append(f'\nUse domain knowledge of {Prompt.Item_type}s to help understand the text and answer the {sent} in the format: For Question i, Yes, {Prompt.Item_type} A and {Prompt.Item_type} B are the same {Prompt.Item_type.lower()}./No, {Prompt.Item_type} A and {Prompt.Item_type} B are different {Prompt.Item_type.lower()}s. For Question i+1, (repeat the above procedures)')
        pair_prompt = '\n'.join(pair_txt)

        prompt = '\n\n'.join([task_description, demo_prompt, pair_prompt])
        return prompt

    @staticmethod
    def generate_prompt_for_single(task_description, demos, pair):
        demo_txts, pair_txts = [], []
        if len(demos) == 1:
            d = demos[0]
            demo_txts.append(
                f'Demonstration:\n{Prompt.Item_type} A is {d[0]}\n{Prompt.Item_type} B is {d[1]}')
            if d[2] == '1':
                demo_txts.append(
                    f'Yes, {Prompt.Item_type} A and {Prompt.Item_type} B are the same {Prompt.Item_type.lower()}.')
            else:
                demo_txts.append(
                    f'No, {Prompt.Item_type} A and {Prompt.Item_type} B are different {Prompt.Item_type.lower()}s.')
        else:
            for i, d in enumerate(demos):
                d = list(d)
                d[0] = d[0].replace('(missing)', '')
                d[1] = d[1].replace('(missing)', '')
                demo_txts.append(
                    f'Demonstration {i+1}:\n{Prompt.Item_type} A is {d[0]}\n{Prompt.Item_type} B is {d[1]}')
                if d[2] == '1':
                    demo_txts.append(
                        f'Yes, {Prompt.Item_type} A and {Prompt.Item_type} B are the same {Prompt.Item_type.lower()}.')
                else:
                    demo_txts.append(
                        f'No, {Prompt.Item_type} A and {Prompt.Item_type} B are different {Prompt.Item_type.lower()}s.')
        demo_prompt = '\n'.join(demo_txts)

        pair_txts.append(
            f'Question:\n{Prompt.Item_type} A is {pair[0]}\n{Prompt.Item_type} B is {pair[1]}')
        pair_txts.append(f'\nUse domain knowledge of {Prompt.Item_type}s to help understand the text and answer the question in the format: Yes, {Prompt.Item_type} A and {Prompt.Item_type} B are the same {Prompt.Item_type.lower()}./No, {Prompt.Item_type} A and {Prompt.Item_type} B are different {Prompt.Item_type.lower()}s.')
        pair_prompt = '\n'.join(pair_txts)

        prompt = '\n\n'.join([task_description, demo_prompt, pair_prompt])
        return prompt

    @staticmethod
    def generate_prompt_to_get_task_descrip(task_description, demos, query_txt, aug=False):
        demos_text = []
        for i, d in enumerate(demos):
            demos_text.append(
                f'Demonstration {i+1}:\n {Prompt.Item_type} A is {d[0]}\n {Prompt.Item_type} B is {d[1]}')
            if d[2] == '1':
                demos_text.append(
                    f'Yes, {Prompt.Item_type} A and {Prompt.Item_type} B are the same.')
            else:
                demos_text.append(
                    f'No, {Prompt.Item_type} A and {Prompt.Item_type} B are different.')
        demos_prompt = '\n'.join(demos_text)

        return '\n'.join([task_description, demos_prompt, query_txt])

    @staticmethod
    def generate_prompt_for_gourp(task_description, demos, pairs, explain=True, remove_noval_keys=True, dataset_name='em-ia'):
        demo_txt = []
        because = 'because... ' if explain else ''

        for i, d in enumerate(demos):
            d = list(d)
            a = d[0].replace('(missing)', '')
            b = d[1].replace('(missing)', '')
            if remove_noval_keys:
                a, b = remov_novel_keys(dataset_name, a, b)
            demo_txt.append(
                f'Demonstration {i+1}:\n{Prompt.Item_type} A is {a}\n{Prompt.Item_type} B is {b}')
            if d[2] == '1':
                demo_txt.append(
                    f'Yes, {Prompt.Item_type} A and {Prompt.Item_type} B are the same {Prompt.Item_type.lower()}.')
            else:
                demo_txt.append(
                    f'No, {Prompt.Item_type} A and {Prompt.Item_type} B are different {Prompt.Item_type.lower()}s.')

        demo_prompt = '\n'.join(demo_txt)

        pair_txt = []

        for i, p in enumerate(pairs):
            p = list(p)
            a = p[0].replace('(missing)', '')
            b = p[1].replace('(missing)', '')
            if remove_noval_keys:
                a, b = remov_novel_keys(dataset_name, a, b)
            pair_txt.append(
                f'Question {i+1}:\n {Prompt.Item_type} A is {a}\n {Prompt.Item_type} B is {b}')

        cnt = len(pairs)
        sent = 'question above' if cnt == 1 else f'above {cnt} questions'
        pair_txt.append(f'\nUse domain knowledge of {Prompt.Item_type}s to help understand the text and answer the {sent} in the format: For Question i, Yes, {Prompt.Item_type} A and {Prompt.Item_type} B are the same {Prompt.Item_type.lower()}./No, {Prompt.Item_type} A and {Prompt.Item_type} B are different {Prompt.Item_type.lower()}s. {because}For Question i+1, (repeat the above procedures)')
        pair_prompt = '\n'.join(pair_txt)

        prompt = '\n\n'.join([task_description, demo_prompt, pair_prompt])
        return prompt


class ERDataset:
    def __init__(self, dataset_name, filter_prop=True, replace=False, sample_num=-1, sample_d_num=-1, join_val_test=False, used=False):

        self.dataset_name = dataset_name
        self.filter_prop = filter_prop
        self.replace = replace
        self.sample_num = sample_num
        self.sample_d_num = sample_d_num
        self.join_val_test = join_val_test
        self.used = used

        self._load_data()
        self._preprocess()

    def _load_data(self):
        train = open_json(f'./data/{self.dataset_name}/train.json')
        test = open_json(f'./data/{self.dataset_name}/test.json')
        valid = open_json(f'./data/{self.dataset_name}/valid.json')

        self.train_t1 = [x[0] for x in train]
        self.train_t2 = [x[1] for x in train]
        self.train_lab = [x[2] for x in train]

        self.test_t1 = [x[0] for x in test]
        self.test_t2 = [x[1] for x in test]
        self.test_lab = [x[2] for x in test]

        self.valid_t1 = [x[0] for x in valid]
        self.valid_t2 = [x[1] for x in valid]
        self.valid_lab = [x[2] for x in valid]

        if self.join_val_test:
            self.test_t1 = self.test_t1+self.valid_t1
            self.test_t2 = self.test_t2+self.valid_t2
            self.test_lab = self.test_lab+self.valid_lab

            self.D_t1 = self.train_t1
            self.D_t2 = self.train_t2
            self.D_lab = self.train_lab
        else:
            self.D_t1 = self.train_t1+self.valid_t1
            self.D_t2 = self.train_t2+self.valid_t2
            self.D_lab = self.train_lab+self.valid_lab

    def _preprocess(self):
        if self.replace:
            self.train_t1 = [filter_str(x[0]) for x in self.train_t1]
            self.train_t2 = [filter_str(x[1]) for x in self.train_t2]

            self.test_t1 = [filter_str(x[0]) for x in self.test_t1]
            self.test_t2 = [filter_str(x[1]) for x in self.test_t2]

            self.valid_t1 = [filter_str(x[0]) for x in self.valid_t1]
            self.valid_t2 = [filter_str(x[1]) for x in self.valid_t2]

        if self.filter_prop:
            for i in range(len(self.D_t1)):
                # print(self.D_t1[i])
                self.D_t1[i] = filtering(self.dataset_name, self.D_t1[i])
                self.D_t2[i] = filtering(self.dataset_name, self.D_t2[i])

            for i in range(len(self.test_t1)):
                self.test_t1[i] = filtering(self.dataset_name, self.test_t1[i])
                self.test_t2[i] = filtering(self.dataset_name, self.test_t2[i])

        if self.sample_num != -1 and self.sample_num < len(self.test_lab):
            pos_ids = [i for i, l in enumerate(self.test_lab) if l == '1']
            neg_ids = [i for i, l in enumerate(self.test_lab) if l == '0']
            pos_samp_num = int(
                (len(pos_ids)/len(self.test_lab))*self.sample_num)
            neg_samp_num = self.sample_num - pos_samp_num
            pos_sam_ids = sample(pos_ids, pos_samp_num)
            neg_sam_ids = sample(neg_ids, neg_samp_num)
            sample_ids = pos_sam_ids+neg_sam_ids
            shuffle(sample_ids)
            self.test_t1 = [self.test_t1[i] for i in sample_ids]
            self.test_t2 = [self.test_t2[i] for i in sample_ids]
            self.test_lab = [self.test_lab[i] for i in sample_ids]

        if self.sample_d_num != -1 and self.sample_d_num < len(self.D_lab):
            pos_ids = [i for i, l in enumerate(self.D_lab) if l == '1']
            neg_ids = [i for i, l in enumerate(self.D_lab) if l == '0']
            pos_samp_num = int(
                (len(pos_ids)/len(self.D_lab))*self.sample_d_num)
            neg_samp_num = self.sample_d_num - pos_samp_num
            pos_sam_ids = sample(pos_ids, pos_samp_num)
            neg_sam_ids = sample(neg_ids, neg_samp_num)
            sample_ids = pos_sam_ids+neg_sam_ids
            shuffle(sample_ids)
            self.D_t1 = [self.D_t1[i] for i in sample_ids]
            self.D_t2 = [self.D_t2[i] for i in sample_ids]
            self.D_lab = [self.D_lab[i] for i in sample_ids]

        self.fill_missing_prop()
        if self.used and self.dataset_name in ['em-ag', 'em-da', 'em-ds']:
            self.merge_props()

    def _fill_missing_prop_helper(self, t1, t2, dn):
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

        for i, t in enumerate(zip(t1, t2)):
            a, b = t[0], t[1]
            ad = json.loads(add_quotation_mark_to_key(dn, '{'+a+'}'))
            bd = json.loads(add_quotation_mark_to_key(dn, '{'+b+'}'))

            for k in keys:
                if k == main_key[dn]:
                    continue
                if k in ad and k in bd:
                    av = ad[k]
                    bv = bd[k]
                    emp1 = (len(av) == 0 or '(missing)' in av)
                    emp2 = (len(bv) == 0 or '(missing)' in bv)
                    if emp1 and emp2:
                        continue

                    if emp1:  # av is empty
                        bv_lis = bv.split(' ')
                        in_ids = []
                        for ind, word in enumerate(bv_lis):
                            if is_float(word):
                                word = str(int(float(word)))
                            if word in ad[main_key[dn]] and len(word) > 2:
                                ad[main_key[dn]] = ad[main_key[dn]
                                                      ].replace(word, '')
                                in_ids.append(ind)
                        av = ' '.join([bv_lis[ind] for ind in in_ids])

                    if emp2:  # bv is empty
                        av_lis = av.split(' ')
                        in_ids = []
                        for ind, word in enumerate(av_lis):
                            if is_float(word):
                                word = str(int(float(word)))
                            if word in bd[main_key[dn]] and len(word) > 2:
                                bd[main_key[dn]] = bd[main_key[dn]
                                                      ].replace(word, '')
                                in_ids.append(ind)
                        bv = ' '.join([av_lis[ind] for ind in in_ids])
                    ad[k] = av
                    bd[k] = bv

            ad[main_key[dn]] = ' '.join(ad[main_key[dn]].split())
            bd[main_key[dn]] = ' '.join(bd[main_key[dn]].split())
            t1[i] = json.dumps(ad).replace(
                '{', '').replace('}', '').replace('(missing)', '')
            t2[i] = json.dumps(bd).replace(
                '{', '').replace('}', '').replace('(missing)', '')

        return t1, t2

    def fill_missing_prop(self):
        self.D_t1, self.D_t2 = self._fill_missing_prop_helper(
            self.D_t1, self.D_t2, self.dataset_name)
        self.test_t1, self.test_t2 = self._fill_missing_prop_helper(
            self.test_t1, self.test_t2, self.dataset_name)

    def merge_helper(self, t1, t2, dn, sep='[SEPRATOR]'):
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

        for i, t in enumerate(zip(t1, t2)):
            a, b = t[0], t[1]
            ad = json.loads(add_quotation_mark_to_key(dn, '{'+a+'}'))
            bd = json.loads(add_quotation_mark_to_key(dn, '{'+b+'}'))
            sa = sep.join(
                [f'{ad[k]}' for k in keys if k in ad and len(ad[k]) > 0])
            sb = sep.join(
                [f'{bd[k]}' for k in keys if k in bd and len(bd[k]) > 0])
            for k in keys:
                if k != main_key[dn]:
                    ad.pop(k)
                    bd.pop(k)
            ad[main_key[dn]] = sa
            bd[main_key[dn]] = sb
            t1[i] = json.dumps(ad).replace(
                '{', '').replace('}', '').replace('(missing)', '')
            t2[i] = json.dumps(bd).replace(
                '{', '').replace('}', '').replace('(missing)', '')

        return t1, t2

    def merge_props(self, sep='[SEPRATOR]'):
        self.D_t1, self.D_t2 = self.merge_helper(
            self.D_t1, self.D_t2, self.dataset_name, sep=sep)
        self.test_t1, self.test_t2 = self.merge_helper(
            self.test_t1, self.test_t2, self.dataset_name, sep=sep)


class GPTPOOL:
    def __init__(self, key_file='keys.txt', model="gpt-3.5-turbo-0301", temp=-1):
        self.key_file = key_file
        self.model = model
        self.temp = temp

    def get_key(self):
        with open(self.key_file, 'r') as f:
            keys = [x.strip() for x in f.readlines()]

        cur_key = copy.deepcopy(keys[0])
        keys = keys[1:]+[cur_key]

        with open(self.key_file, 'w') as f:
            f.write('\n'.join(keys))

        self.cur_key = cur_key

        return cur_key

    def query(self, ask, get_lower=True):
        key = self.get_key()
        print(f'cur_key: {key}')
        os.environ["OPENAI_API_KEY"] = key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[{
                "role": "system", "content": "Assistant is an intellectual chatbot designed to follow you instructions."
            }, {"role": "user", "content": ask}],
            temperature=self.temp if self.temp != -1 else 1,
        )
        ans = completion.choices[0].message['content']
        if get_lower:
            ans = ans.lower().strip().replace('\n', ' ').replace('  ', ' ')
        else:
            ans = ans.strip().replace('\n', ' ').replace('  ', ' ')
        return ans


class SinglePrompt:
    def __init__(self, gpt: GPTPOOL, data: ERDataset, root, model_name, predefined_demos, task_description):

        self.gpt = gpt
        self.data = data
        self.save_root = f'./{root}/{data.dataset_name}/'
        mkdir(self.save_root)
        self.logger = logger_config(
            log_path=f'{self.save_root}/log.txt', logging_name=f'SinglePrompt')
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.predefined_demos = predefined_demos
        self.task_description = task_description

    def run(self):
        demos4pair = []
        for _ in range(len(self.data.test_lab)):
            demos4pair.append(self.predefined_demos)
        self.logger.info(
            f'unique demos: {len(set([d for demos in demos4pair for d in demos]))}')
        self.logger.info(
            f'tol_demo_num: {sum([len(demos) for demos in demos4pair])}')
        self.query_dataset(self.data, demos4pair, self.task_description)

    def query_dataset(self, data: ERDataset, demos4pair, task_description):
        fail2reply, false_reply, tol_records = [], [], []
        fn_cnt, fp_cnt, tol_cnt = 0, 0, 0
        for pid in range(len(data.test_lab)):
            self.logger.info(
                f'------------------- query pair [{pid+1}/{len(data.test_lab)}] -------------------')
            pairs = [self.data.test_t1[pid], self.data.test_t2[pid]]
            demos = demos4pair[pid]
            demos = [[self.data.D_t1[i], self.data.D_t2[i],
                      self.data.D_lab[i]] for i in demos]
            ask = Prompt.generate_prompt_for_single(
                task_description, demos, pairs)
            query_cnt = 0
            while True:
                if query_cnt > 5:
                    break
                try:
                    self.logger.info(ask)
                    self.logger.info(
                        f'******************** {fp_cnt+fn_cnt}/{tol_cnt}, fp: {fp_cnt}, fn: {fn_cnt} ******************')
                    ans = self.gpt.query(ask, get_lower=False)
                    self.logger.info(ans)
                except:
                    self.logger.info(
                        '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! err! [GPT fails to replay] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    record = {
                        'ask': ask,
                        'pid': pid,
                        'demo': demos4pair[pid],
                        'ask_tok': tokens_num(self.encoding, ask),
                    }
                    fail2reply.append(record)
                    save_json(
                        fail2reply, f'./{self.save_root}/fail2reply.json')
                    time.sleep(random.randint(8, 10))
                    continue

                ans = ans.replace('No', 'no').replace('Yes', 'yes').replace(
                    'Question', 'question').replace('QUESTION', 'question')+'.'
                yf = ans.count('yes.') + ans.count('yes,')
                nf = ans.count('no.') + ans.count('no,')
                try:
                    assert yf+nf == 1
                except:
                    self.logger.info(
                        f'!!!!!!!!!!!!!!!!!!!!!!!!!!!! err! [yf: {yf}, nf: {nf} wrong numbers] !!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    record = {
                        'ask': ask,
                        'ans': ans,
                        'pid': pid,
                        'demo': demos4pair[pid],
                        'yf': yf,
                        'nf': nf,
                        'ask_tok': tokens_num(self.encoding, ask),
                        'ans_tok': tokens_num(self.encoding, ans),
                    }
                    false_reply.append(record)
                    save_json(
                        false_reply, f'./{self.save_root}/false_reply.json')
                    query_cnt += 1
                    continue
                break

            if query_cnt > 5:
                continue

            curi = 0
            preds, inds, ans4pair = [], [], []
            while True:
                if ans[curi] == 'y' and (ans[curi: curi+4] == 'yes.' or ans[curi: curi+4] == 'yes,'):
                    preds.append(1)
                    inds.append(curi)
                if ans[curi] == 'n' and (ans[curi: curi+3] == 'no.' or ans[curi: curi+3] == 'no,'):
                    preds.append(0)
                    inds.append(curi)
                curi += 1
                if curi >= len(ans) or len(preds) >= 1:
                    break

            inds.append(len(ans))
            for i in range(len(inds)-1):
                ans4pair.append(ans[inds[i]: inds[i+1]])

            for i, pair in enumerate([pid]):
                if preds[i] != int(self.data.test_lab[pair]):
                    self.logger.info(
                        f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! false prediction! [pred: {preds[i]}, lab: {self.data.test_lab[pair]}] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    self.logger.info(f'Question {i+1}:')
                    self.logger.info(self.data.test_t1[pair])
                    self.logger.info(self.data.test_t2[pair])
                    self.logger.info(ans4pair[i])
                    self.logger.info(
                        f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    if preds[i] == 1:
                        fp_cnt += 1
                    else:
                        fn_cnt += 1

            tol_cnt += 1

            record = {
                'ask': ask,
                'ans': ans,
                'pid': pid,
                'demo': demos4pair[pid],
                'ask_tok': tokens_num(self.encoding, ask),
                'ans_tok': tokens_num(self.encoding, ans),
                'preds': preds,
                'labs': [int(self.data.test_lab[pid])]
            }
            tol_records.append(record)
            save_json(tol_records, f'./{self.save_root}/tol_records.json')

        result = self._cal_results(tol_records)
        save_json(result, f'./{self.save_root}/result.json')
        self.logger.info(
            f'============================= result =============================')
        self.logger.info(result)
        self.logger.info(
            f'=====================================================================================')
        return result

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
