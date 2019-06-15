from pathlib import Path
import numpy as np
import torch
from pyknp import Juman
import sys
sys.path.append("../../../pytorch-pretrained-BERT/")
from pytorch_pretrained_bert import BertTokenizer, BertModel  # noqa


class JumanTokenizer():

    def __init__(self):
        self.juman = Juman()

    def tokenize(self, text):
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]


class BertWithJumanModel():

    def __init__(self, bert_path, vocab_file_name="vocab.txt", use_cuda=False):
        # 京大の形態素解析器
        self.juman_tokenizer = JumanTokenizer()

        # 訓練済みモデル(configの設定はどこ？)
        # モデルのロードの仕方には2種類ある？
        self.model = BertModel.from_pretrained(bert_path)

        # Bertの形態素解析器 do_lower_caseとかの意味
        # do_basic_tokenize=Falseは必須。
        self.bert_tokenizer = BertTokenizer(Path(bert_path) / vocab_file_name,
                                            do_lower_case=False, do_basic_tokenize=False)
        self.use_cuda = use_cuda

    def _preprocess_text(self, text):
        return text.replace(" ", "")  # for Juman

    def get_sentence_embedding(self, text, pooling_layer=-2, pooling_strategy="REDUCE_MEAN"):

        preprocessed_text = self._preprocess_text(text)

        # Juman++で形態素解析を行う。
        tokens = self.juman_tokenizer.tokenize(preprocessed_text)

        # Jumanを通したあとBertのトークナイザを通す。
        # Bertに登録されてないトークンは[UKN]に置換される。
        # Bertのトークナイザは空白区切をするだけ。
        # Bertのトークナイザはサブワード分割もする。
        bert_tokens = self.bert_tokenizer.tokenize(" ".join(tokens))

        ids = self.bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + bert_tokens[:126] + ["[SEP]"])  # max_seq_len-2
        tokens_tensor = torch.tensor(ids).reshape(1, -1)

        if self.use_cuda:
            tokens_tensor = tokens_tensor.to('cuda')
            self.model.to('cuda')

        self.model.eval()
        with torch.no_grad():
            # 12層の隠れ層を取り出す。
            all_encoder_layers, _ = self.model(tokens_tensor)
        # assert(12 == len(all_encoder_layers))

        # 適当な層を取り出す。[トークン数,次元数]
        embedding = all_encoder_layers[pooling_layer].cpu().numpy()[0]

        # トークン数の軸に沿って...をする。
        if pooling_strategy == "REDUCE_MEAN":
            return np.mean(embedding, axis=0)
        elif pooling_strategy == "REDUCE_MAX":
            return np.max(embedding, axis=0)
        elif pooling_strategy == "REDUCE_MEAN_MAX":
            return np.r_[np.max(embedding, axis=0), np.mean(embedding, axis=0)]
        elif pooling_strategy == "CLS_TOKEN":
            return embedding[0]
        else:
            raise ValueError("specify valid pooling_strategy: {REDUCE_MEAN, REDUCE_MAX, REDUCE_MEAN_MAX, CLS_TOKEN}")
