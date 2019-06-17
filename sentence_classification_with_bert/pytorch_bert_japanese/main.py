#!/usr/bin/env python
# -*- coding: utf-8 -*-
import bert_juman

DIR_PATH = "/home/ubuntu/data/kyodai_bert/Japanese_L-12_H-768_A-12_E-30_BPE"


def sample_0():
    print("now loading model...")
    bert = bert_juman.BertWithJumanModel(DIR_PATH, use_cuda=True)
    print("loading done")

    v = bert.get_sentence_embedding("吾 輩は猫である。")
    print(v[:10])
    # print("---")
    # v = bert.get_sentence_embedding("これは何ですか？")
    # print("---")
    # v = bert.get_sentence_embedding("メソッドを呼ぶことで様々な処理が可能となる。")


def sample_1():
    tokenizer = bert_juman.JumanTokenizer()
    tokens = tokenizer.tokenize("#")
    print(tokens)


if __name__ == "__main__":
    sample_1()
