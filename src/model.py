from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer, DebertaTokenizer, XLNetTokenizer, DistilBertTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import torch.nn.functional as F
import json
from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, DistilBertModel, RobertaModel, AutoModel, DebertaModel, XLNetModel
from sentence_transformers import SentenceTransformer
from fastbm25 import fastbm25
# from tutel import moe as tutel_moe

"""Params."""


class param:
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    cls = 1
    encoder_path = "encoder.pt"
    moe_path = "moe.pt"
    cls_path = "cls.pt"
    model_root = "checkpoint"
    num_labels = 2
    hidden_size = 768


class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(
            'bert-base-multilingual-cased')

    def forward(self, x, mask=None, segment=None):
        outputs = self.encoder(x, attention_mask=mask, token_type_ids=segment)
        if param.cls:
            feat = outputs[1]
        else:
            token_len = outputs[0].shape[1]
            feat = torch.sum(outputs[0], dim=1)
            feat = torch.div(feat, token_len)
        return feat


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class MPEncoder(nn.Module):
    def __init__(self):
        super(MPEncoder, self).__init__()
        model_path = "all-mpnet-base-v2"
        self.encoder = AutoModel.from_pretrained(model_path)

    def forward(self, x, mask=None, segment=None):
        inp = {'input_ids': x.detach(), 'attention_mask': mask.detach()}
        outputs = self.encoder(**inp)
        if param.cls:
            feat = outputs[1]
        else:
            feat = mean_pooling(outputs, mask.detach())
            feat = F.normalize(feat, p=2, dim=1)
        return feat


class DistilBertEncoder(nn.Module):
    def __init__(self):
        super(DistilBertEncoder, self).__init__()
        self.encoder = DistilBertModel.from_pretrained(
            'distilbert-base-multilingual-cased')
        self.pooler = nn.Linear(param.hidden_size, param.hidden_size)

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        feat = self.pooler(pooled_output)
        return feat


class RobertaEncoder(nn.Module):
    def __init__(self):
        super(RobertaEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-base')

    def forward(self, x, mask=None, segment=None):
        outputs = self.encoder(x, attention_mask=mask)
        sequence_output = outputs[0]
        feat = sequence_output[:, 0, :]
        return feat


class DistilRobertaEncoder(nn.Module):
    def __init__(self):
        super(DistilRobertaEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained('distilroberta-base')
        self.pooler = nn.Linear(param.hidden_size, param.hidden_size)

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        sequence_output = outputs[0]
        feat = sequence_output[:, 0, :]
        return feat


class XLNetEncoder(nn.Module):
    def __init__(self):
        super(XLNetEncoder, self).__init__()
        self.encoder = XLNetModel.from_pretrained("xlnet-base-cased")

    def forward(self, x, mask=None, segment=None):
        outputs = self.encoder(x, attention_mask=mask, token_type_ids=segment)
        if param.cls:
            feat = outputs.last_hidden_state
            feat = feat[:, 0, :]
        else:
            token_len = outputs[0].shape[1]
            feat = torch.sum(outputs[0], dim=1)
            feat = torch.div(feat, token_len)
        return feat


class DebertaBaseEncoder(nn.Module):
    def __init__(self):
        super(DebertaBaseEncoder, self).__init__()
        self.encoder = DebertaModel.from_pretrained("deberta-base")

    def forward(self, x, mask=None, segment=None):
        outputs = self.encoder(x, attention_mask=mask, token_type_ids=segment)
        if param.cls:
            feat = outputs.last_hidden_state
            feat = feat[:, 0, :]
        else:
            token_len = outputs[0].shape[1]
            feat = torch.sum(outputs[0], dim=1)
            feat = torch.div(feat, token_len)
        return feat


class DebertaLargeEncoder(nn.Module):
    def __init__(self):
        super(DebertaLargeEncoder, self).__init__()
        self.encoder = DebertaModel.from_pretrained("deberta-large")

    def forward(self, x, mask=None, segment=None):
        outputs = self.encoder(x, attention_mask=mask, token_type_ids=segment)
        if param.cls:
            feat = outputs.last_hidden_state
            feat = feat[:, 0, :]
        else:
            token_len = outputs[0].shape[1]
            feat = torch.sum(outputs[0], dim=1)
            feat = torch.div(feat, token_len)
        return feat


class Classifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(param.hidden_size, param.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, x):
        x = self.dropout(x)
        out = self.classifier(x)
        return out

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class MOEClassifier(nn.Module):
    def __init__(self, units, dropout=0.1):
        super(MOEClassifier, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(units, param.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, x):
        x = self.dropout(x)
        out = self.classifier(x)
        return out

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def make_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, exm_id, task_id=-1):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.exm_id = exm_id
        self.task_id = task_id


def convert_one_example_to_features(tuple, max_seq_length, cls_token, sep_token, pad_token, tokenizer):
    tokens = tokenizer.tokenize(tuple)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]
    tokens = [cls_token] + tokens + [sep_token]
    segment_ids = [0]*(len(tokens))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([0] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def convert_one_example_to_features_sep(tuple, max_seq_length, cls_token, sep_token, pad_token, tokenizer):
    left = tuple.split(sep_token)[0]
    right = tuple.split(sep_token)[1]
    ltokens = tokenizer.tokenize(left)
    rtokens = tokenizer.tokenize(right)
    more = len(ltokens) + len(rtokens) - max_seq_length + 3
    if more > 0:
        if more < len(rtokens):
            rtokens = rtokens[:(len(rtokens) - more)]
        elif more < len(ltokens):
            ltokens = ltokens[:(len(ltokens) - more)]
        else:
            rtokens = rtokens[:50]
            ltokens = ltokens[:50]
    tokens = [cls_token]+ltokens+[sep_token] + rtokens+[sep_token]
    segment_ids = [0]*(len(ltokens)+2) + [1]*(len(rtokens)+1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([0] * padding_length)

    return input_ids, input_mask, segment_ids


def convert_examples_to_features(text=None, labels=None, max_seq_length=128, tokenizer=None,
                                 cls_token="[CLS]", sep_token='[SEP]',
                                 pad_token=0, task_ids=None):
    features = []
    if labels == None:
        labels = [0] * len(text)
    for ex_index, pair in enumerate(text):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(text)))
        if 1:
            fea_pair = []
            for i, tuple in enumerate(pair):
                if sep_token in tuple:
                    input_ids, input_mask, segment_ids = convert_one_example_to_features_sep(
                        tuple, max_seq_length, cls_token, sep_token, pad_token, tokenizer)
                else:
                    input_ids, input_mask, segment_ids = convert_one_example_to_features(
                        tuple, max_seq_length, cls_token, sep_token, pad_token, tokenizer)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                if task_ids:
                    fea_pair.append(
                        InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=labels[ex_index],
                                      exm_id=ex_index,
                                      task_id=task_ids[ex_index]))
                else:
                    fea_pair.append(
                        InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=labels[ex_index],
                                      exm_id=ex_index,
                                      task_id=-1))
        else:
            continue
        features.append(fea_pair)

    return features


def convert_fea_to_tensor00(features_list, batch_size, do_train):
    features = [x[0] for x in features_list]

    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    all_exm_ids = torch.tensor([f.exm_id for f in features], dtype=torch.long)
    all_task_ids = torch.tensor(
        [f.task_id for f in features], dtype=torch.long)

    # tuple: (tensor1, tensor2,...) tensor_i=torch.Size([567, 128]),对应每个item（ids、mask、...)
    # (tensor([[  101, 52586, 37925,  ...,     0,     0,     0],        [  101,   100, 60282,  ...,     0,     0,     0],        [  101,   100, 15630,  ...,     0,     0,     0],        ...,        [  101, 10467, 55085,  ...,     0,     0,     0],        [  101,   100, 11322,  ...,     0,     0,     0],        [  101, 11357, 61408,  ...,     0,     0,     0]]),
    # tensor([[1, 1, 1,  ....0, 0, 0]]),
    # tensor([[0, 0, 0,  ....0, 0, 0]]),
    # tensor([0, 0, 0, 0, ... 0, 0, 0]),
    # tensor([  0,   1,   ...565, 566]),
    # tensor([-1, -1, -1, ..., -1, -1]))
    dataset = TensorDataset(all_input_ids, all_input_mask,
                            all_segment_ids, all_label_ids, all_exm_ids, all_task_ids)

    if do_train == 0:
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=batch_size)
    else:
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler,
                                batch_size=batch_size, drop_last=True)
    return dataloader


def convert_text_to_tensor(texts, encoder, tokenizer, batch_size=32, do_train=0):

    fea = convert_examples_to_features(text=texts, labels=None, max_seq_length=128, tokenizer=tokenizer,
                                       cls_token="[CLS]", sep_token='[SEP]',
                                       pad_token=0, task_ids=None)

    data_loader = convert_fea_to_tensor00(
        features_list=fea, batch_size=batch_size, do_train=do_train)

    encoder.eval()

    sample_tensor = None
    for (reviews, mask, segment, labels, exm_id, task_id) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)
        labels = make_cuda(labels)
        truelen = torch.sum(mask, dim=1)

        with torch.no_grad():
            feat = encoder(reviews, mask, segment)
            sample_tensor = feat if sample_tensor is None else torch.cat(
                (sample_tensor, feat), 0)

    return sample_tensor


def convert_text_to_tensor_roberta(texts, encoder, tokenizer, batch_size=32, do_train=0):

    fea = convert_examples_to_features_roberta(pairs=texts, labels=None, max_seq_length=128, tokenizer=tokenizer,
                                               pad_token=0)

    data_loader = convert_fea_to_tensor00(
        features_list=fea, batch_size=batch_size, do_train=do_train)

    encoder.eval()

    sample_tensor = None
    for (reviews, mask, segment, labels, exm_id, task_id) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)
        labels = make_cuda(labels)
        truelen = torch.sum(mask, dim=1)

        with torch.no_grad():
            feat = encoder(reviews, mask, segment)
            sample_tensor = feat if sample_tensor is None else torch.cat(
                (sample_tensor, feat), 0)

    return sample_tensor


def convert_examples_to_features_roberta(pairs=None, labels=None, max_seq_length=128, tokenizer=None,
                                         cls_token="<s>", sep_token='</s>',
                                         pad_token=0):
    features = []
    if labels == None:
        labels = [0] * len(pairs)
    for ex_index, pair in enumerate(pairs):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(pairs)))
        if 1:
            fea_pair = []
            for i, tuple in enumerate(pair):
                if sep_token in tuple:
                    input_ids, input_mask, segment_ids = convert_one_example_to_features_roberta_sep(
                        tuple, max_seq_length, cls_token, sep_token, pad_token, tokenizer)
                else:
                    input_ids, input_mask, segment_ids = convert_one_example_to_features(
                        tuple, max_seq_length, cls_token, sep_token, pad_token, tokenizer)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                fea_pair.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=labels[ex_index],
                                  exm_id=ex_index))
        else:
            continue
        features.append(fea_pair)
    return features


def convert_one_example_to_features_roberta_sep(tuple, max_seq_length, cls_token, sep_token, pad_token, tokenizer):
    left = tuple.split(sep_token)[0]
    right = tuple.split(sep_token)[1]
    ltokens = tokenizer.tokenize(left)
    rtokens = tokenizer.tokenize(right)
    more = len(ltokens) + len(rtokens) - max_seq_length + 4
    if more > 0:
        if more < len(rtokens):
            rtokens = rtokens[:(len(rtokens) - more)]
        elif more < len(ltokens):
            ltokens = ltokens[:(len(ltokens) - more)]
        else:
            rtokens = rtokens[:50]
            ltokens = ltokens[:50]
    tokens = [cls_token] + ltokens + [sep_token] + \
        [sep_token] + rtokens + [sep_token]
    segment_ids = [0]*(len(ltokens)+2) + [1]*(len(rtokens)+1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([0] * padding_length)

    return input_ids, input_mask, segment_ids


def get_text_sim(text1, text2, encoder, tokenizer):
    ts1 = convert_text_to_tensor(
        text1, encoder, tokenizer, batch_size=32, do_train=0)
    ts2 = convert_text_to_tensor(
        text2, encoder, tokenizer, batch_size=32, do_train=0)
    sim = []
    for t1, t2 in zip(ts1, ts2):
        cos_sim = F.cosine_similarity(t1, t2, dim=0)
        sim.append(cos_sim.tolist())
    return sim


def open_json(path):
    with open(path, "r", encoding='ISO-8859-1') as f:
        data = json.load(f)
    return data


def get_metric(pred, lab):
    acc = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    p = 0
    for i in range(len(lab)):
        if lab[i] == pred[i]:
            acc += 1

        if lab[i] == 1:
            p += 1
            if pred[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred[i] == 1:
                fp += 1
            else:
                tn += 1

    div_safe = 0.000001
    print(
        f'p: {p}, \t n: {len(lab)-p}, \t tp: {tp}, \t fp: {fp}, \t tn: {tn}, \t fn: {fn}')
    recall = tp/(p+div_safe)

    precision = tp/(tp+fp+div_safe)
    f1 = 2*recall*precision/(recall + precision + div_safe)

    acc /= len(pred)

    # print("recall",recall)
    # print("precision",precision)
    # print("f1",f1)

    return f1, recall, acc, p, tp, fp, len(lab)-p, tn, fn


class EncoderModel:
    def __init__(self, encoder_name='BERT', instruction=None):
        self.encoder_name = encoder_name
        if encoder_name == 'BERT':
            self.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-multilingual-cased')
            self.encoder = BertEncoder()
        elif encoder_name == 'deberta-large':
            self.encoder = DebertaLargeEncoder()
            self.tokenizer = DebertaTokenizer.from_pretrained('deberta-large')
        elif encoder_name == 'SBERT':
            self.encoder = SentenceTransformer(
                'sentence-transformers/bert-base-nli-mean-tokens')
            self.tokenizer = None
        elif encoder_name == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.encoder = RobertaEncoder()
        elif encoder_name == 'deberta-base':
            self.encoder = DebertaBaseEncoder()
            self.tokenizer = DebertaTokenizer.from_pretrained('deberta-base')

    def encode(self, texts):
        if self.tokenizer == None:
            if self.encoder_name == 'SBERT':
                return self.encoder.encode(texts)
        else:
            if 'roberta' in self.encoder_name:
                ts = convert_text_to_tensor_roberta(
                    texts, self.encoder, self.tokenizer, batch_size=32, do_train=0)
            else:
                ts = convert_text_to_tensor(
                    texts, self.encoder, self.tokenizer, batch_size=32, do_train=0)
            return ts.numpy()


class BM25:
    def __init__(self, corpus):
        tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
        self.model = fastbm25(tokenized_corpus)

    def sims(self, texta, textb):
        return self.model.similarity_bm25(texta.lower().split(" "), textb.lower().split(" "))
