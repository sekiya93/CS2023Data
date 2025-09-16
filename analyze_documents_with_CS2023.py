#!/usr/bin/env python
############################################################
#
# analyze_documents_with_CS2023.py 
#
# - Update: 2025-09-16 / Create: 2025-09-16
#
# - LDA (Latent Dirichlet Allocation) を用いて，文書群を分析する
# - CS2023 の Knowledge Area に対応させる
# - LDA の実行には，blei-lda (https://github.com/blei-lab/lda-c) を使用する
#
############################################################
#
# - Usage: ./analyze_documents_with_CS2023.py --target target_dir
#            [--settings settings.txt]
#            [--model model]
#            [--word word]
#
# - 必要なライブラリ
# -- (python3 の実行環境)
# -- pandas
# -- nltk
# --- nltk のデータ (wordnet, punkt, averaged_perceptron_tagger, stopwords) も読み込むこと
#
# - 本スクリプトと同じディレクトリ内に，以下のファイルを配置すること
# -- (本スクリプト analyze_documents_with_CS2023.py)
# -- lda (LDA の実行ファイル)
# --- blei-lda (https://github.com/blei-lab/lda-c) をコンパイルして得られる
# -- settings.conf (LDA の設定ファイル)
# -- final.model.text (ssLDA のモデルファイル)
# -- word.csv (LDA で用いる単語リスト)
#
# - 分析対象のテキストファイルを含むディレクトリを指定して実行する
# -- ファイル名は，.txt で終わるファイルとする
# -- 英語で書かれたテキストファイルを想定

############################################################
# 初期設定
############################################################
import argparse
import glob
import logging
from nltk import stem, tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import os
import pandas as pd
import re
import subprocess



# 単語に相応しくない記号等を取り除くパターン
P_S = re.compile(r'^[^a-zA-Z\d]+')
P_E = re.compile(r'[^a-zA-Z\d]+$')

# 数字と記号のみ
P_NUM = re.compile(r'^[\d!@#\$%\^&\*\(\)_\-\+=|\\~\`{}\[\]:;\"\',\.\/\?<>]+$')

# ファイル名
P_TEXT  = re.compile(r'(\.txt)$')
WORD_DATA_SUFFIX = '_words.dat'

import sys
progname = sys.argv[0]

# 作業ディレクトリのフルパスを求めておく
work_dir = os.path.dirname(os.path.abspath(progname))

# 例外処理
# - 恣意的過ぎるか?
SPECIAL_WORDS = ['cs', 'css', 'vs']

# LDA
LDA = f'{work_dir}/lda'
LDA_SETTINGS = f'{work_dir}/settings.conf'
LDA_MODEL = f'{work_dir}/final.model.text'

# - LDA で用いる単語リスト
WORD_FILE = f'{work_dir}/word.csv'
# - LDA で用いる文書-単語情報
DOC_TERM_FILE = 'doc_term.dat'
# - LDA の推論結果の読み込み
LDA_INF_GAMMA_FILE = 'inf-gamma.dat'

# - CS2023 の Knowledge Areaに対応させる
# (べた書きするのは良くないか？)
CS2023_KA_COL = [
    'AI', 'AL', 'AR', 'DM', 'FPL', 'GIT', 
    'HCI', 'MSF', 'NC', 'OS', 'PDC', 'SDF', 
    'SE', 'SEC', 'SEP', 'SF', 'SPD'
]

######################################################
# ログ
######################################################
# - カレントディレクトリに出力することに注意
p_py = re.compile(r'(\.py)$')
log_file = f'{work_dir}/{os.path.basename(p_py.sub('.log', sys.argv[0]))}'
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)
print(f'The log message will be written to {log_file}.', file=sys.stderr)

######################################################
# 引数
######################################################
#            --target text_dir
#            --settings settings.txt
#            --model model   
#            --word word
parser = argparse.ArgumentParser(prog=progname)
parser.add_argument('--target', type=str) # 分析対象のテキストファイルを含むディレクトリ
parser.add_argument('--settings', type=str, default=LDA_SETTINGS) # settings.txt
parser.add_argument('--model', type=str, default=LDA_MODEL) # model
parser.add_argument('--word', type=str, default=WORD_FILE) # word

args = parser.parse_args()
logging.info(f'args: {args}')

######################################################
# テキストの処理
######################################################
# https://stackoverflow.com/questions/61982023/using-wordnetlemmatizer-lemmatize-with-pos-tags-throws-keyerror より
# - pos_tag() の tag を wordnet の tag に変換
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# Lemmatisation
lemmatizer = stem.WordNetLemmatizer()

# Stop words
STOPWORD_SET = set(stopwords.words('english'))

def text_to_words(text_file):
    all_words, rejected_words = {}, {}
    
    nb_line = 0 
    f = open(text_file, 'r')
    for org_line in f:
        # 行番号
        nb_line += 1
        # スラッシュを空白に置換
        line = org_line.replace('/', ' ')

        for word, pos in pos_tag(tokenize.word_tokenize(line.lower(), language='english')):
            # print(word, pos, file=sys.stderr)
            # pprint(pos)
            # 単語の前後の記号などを除去
            word = P_S.sub('', word)
            word = P_E.sub('', word)

            # サイズが 2未満
            # - 空文字列や1文字の語はこの時点で除外
            # - C言語，O記法が除外されるが...
            # if len(word) < 2:
            #     continue
            
            # 数字や記号のみ
            if P_NUM.match(word):
                continue

            # 未知語として除外済みの単語
            # - ログも出力しない
            if word in rejected_words:
                continue
            
            # 未知語を除去
            # - CS 関連の用語は wordnet に含まれていないことがあるので，除外しないこととする (2024-06-04)
            # if not wordnet.synsets(word):
            #     logging.warning("Unknown word ({0}) in {1}. Skip.".format(word, text_file))
            #     rejected_words[word] = 1
            #     continue
            
            if word not in SPECIAL_WORDS:
                # 語幹化
                wordnet_pos = get_wordnet_pos(pos)
                if wordnet_pos is not None:
                    l_word = lemmatizer.lemmatize(word, wordnet_pos)
                else:
                    l_word = lemmatizer.lemmatize(word)

                if word != l_word:
                    # logging.debug(f'[{text_file}:{nb_line:04d}] lemmatize: {l_word} <- {word}({pos}/{wordnet_pos})')
                    word = l_word
            
                # (改めて)未知語として除外済みの単語
                # - ログも出力しない
                if word in rejected_words:
                    continue

                # 短過ぎる単語
                # - CS で意味のある単語が除外されそうなので止める (2024-06-04)
                #
                if len(word) < 1:
                    logging.warning(f'[{text_file}:{nb_line:04d}] Too short word ({word}). Skip.')
                    rejected_words[word] = 1
                    continue

            # ストップワード
            if word in STOPWORD_SET:
                logging.warning(f'[{text_file}:{nb_line:04d}] Stop word ({word}). Skip.')
                rejected_words[word] = 1
                continue

            # 出現頻度情報の更新
            if word not in all_words:
                all_words[word] = 0
            all_words[word] += 1 

    f.close()

    return all_words, rejected_words

############################################################
# 単語情報の読み込み
############################################################
WORD_ID_MAP = dict()
P_ID_WORD = re.compile(r'^(\d+):(.+)')
def read_word_file(word_file):
    logging.info(f'Reading word_file "{word_file}"...')
    nb_line = 0
    nb_word = 0
    with open(word_file, 'rt') as fin:
        for line in fin:
            nb_line += 1
            m = P_ID_WORD.search(line)
            if not m:
                logging.warning(f'{word_file}:{nb_line:04d} Cannot extract word data from "{line}". Skip.')
                continue
            word_id, word = int(m.group(1))-1, m.group(2)
            WORD_ID_MAP[word] = word_id 
            nb_word += 1

    logging.info(f'Number of word: {nb_word} / Number of line: {nb_line}')

############################################################
# LDA 
############################################################
# (s)sLDA のモデル情報を，blei-lda で用いるために変換
# - final.beta, final.other の 2つのファイルを生成
# -- final.beta: 各トピックにおける各単語の出現確率
# -- final.other: トピック数，語彙数，alpha
# - (最初から final.beta, final.other を用意しておいても良いかも)
LDA_FINAL_BETA, LDA_FINAL_OTHER = 'final.beta', 'final.other'
P_ALPHA = re.compile(r'^alpha:\s+([\d\.]+)')
P_TOPICS = re.compile(r'^number\s+of\s+topics:\s+([\d\.]+)')
P_TERMS = re.compile(r'^size\s+of\s+vocab:\s+([\d\.]+)')

def set_up_lda_model(target_dir, sslda_model_file):

    logging.info(f'Reading {sslda_model_file}')

    # 出力先の準備
    beta_text = ''

    # 読み込み
    with open(sslda_model_file, 'rt') as fin:
        in_beta = False
        for line in fin:
            if in_beta:
                if line[:5] == 'etas:':
                    break
                else:
                    beta_text += line
            elif line[:6] == 'betas:':
                in_beta = True
                continue
            else:
                m_al = P_ALPHA.search(line)
                if m_al:
                    alpha = m_al.group(1)
                    continue
                m_to = P_TOPICS.search(line)
                if m_to:
                    num_topics = m_to.group(1)
                    continue
                m_te = P_TERMS.search(line)
                if m_te:
                    num_terms = m_te.group(1)

    # LDA_FINAL_BETA の出力
    beta_file = f'{target_dir}/{LDA_FINAL_BETA}'
    with open(beta_file, 'wt') as fout:
        fout.write(beta_text)
    logging.info(f'Create {LDA_FINAL_BETA} ({beta_file}).')

    # LDA_FINAL_OTHER の出力
    other_file = f'{target_dir}/{LDA_FINAL_OTHER}'
    with open(other_file, 'wt') as fout:
        fout.write(f'num_topics {num_topics}\nnum_terms {num_terms}\nalpha {alpha}\n')
    logging.info(f'Create {LDA_FINAL_OTHER} ({other_file}).')

# LDA の推論結果 (inf-gamma.dat) の読み込み
def read_inf_gamma_file(target_dir):
    ka_data = {ka: [] for ka in CS2023_KA_COL}
    gamma_file = f'{target_dir}/{LDA_INF_GAMMA_FILE}'
    with open(gamma_file, 'rt') as fin:
        for line in fin:
            value_col = [float(s) for s in line.strip().split(' ')]
            sum_gamma = sum(value_col)
            for ka, value in zip(CS2023_KA_COL, value_col):
                ka_data[ka].append(value / sum_gamma)
    
    return pd.DataFrame(ka_data)

############################################################
# メイン
############################################################
if not args.target:
    print('Error: --target is required.', file=sys.stderr)
    sys.exit(1)

read_word_file(args.word)

set_up_lda_model(args.target, args.model)

text_file_to_words_map = {} # 一時保存
df_words = {} # IDF を算出するため利用
num_of_text = 0

text_file_col = glob.glob(args.target + '/*.txt')

for text_file in text_file_col:
    all_words, rejected_words = text_to_words(text_file)
    if len(all_words) < 1:
        logging.warning(f'[{text_file}] Cannot extract any words. Skip.')
        continue

    # 単語情報のログ出力
    logging.info(f'[{text_file}] Number of extracted words: {len(all_words)}. Rejected words: {len(rejected_words)}.')
    logging.debug(f'[{text_file}] Extracted words: {all_words}.')
    logging.debug(f'[{text_file}] Rejected words: {rejected_words}.')

    # 文書数を数え上げ
    num_of_text += 1
    
    # DF を数え上げ
    for word in all_words.keys():
        if word not in df_words:
            df_words[word] = 0
        df_words[word] += 1
                                                                                
    # 出力する前に単語情報を一時保持
    text_file_to_words_map[text_file] = all_words

# 前処理完了    
logging.info(f'Number of Text files: {num_of_text}.')

# 単語頻度情報ファイルの書き出し
columns = ['text_file', 'total_term_freq', 'nb_of_terms', 'word_data', 'full_word_data']
all_text_file_word_data = {k: [] for k in columns}

for text_file in text_file_col:    
    words_file = P_TEXT.sub(WORD_DATA_SUFFIX, text_file)
    # logging.info("Generate the words file ({0}).".format(words_file))
    logging.info(f'[{text_file}] Generate the words file {words_file}.')

    if text_file in text_file_to_words_map:
        all_words = text_file_to_words_map[text_file]
    else:
        continue
    
    f = open(words_file, 'w')
    word_data = []
    full_word_data = []
    for word in all_words.keys():
        # word_id
        if word not in WORD_ID_MAP:
            logging.warning(f'[{text_file}] There is no "{word}" in the word file "{args.word}". Skip.')
            continue
        word_id = WORD_ID_MAP[word]
        # logging.debug(f'[{text_file}] {word_id}:{word} (tf, df, tf_idf) = ({tf}, {df}, {tf_idf})')
        tf = all_words[word]
        f.write(f'{word_id}:{word}:{tf}\n')
        word_data.append(f'{word_id}:{tf}')
        full_word_data.append(f'{word_id}:{word}:{tf}')
    
    f.close()

    all_text_file_word_data['text_file'].append(os.path.basename(text_file))
    all_text_file_word_data['total_term_freq'].append(sum(all_words.values()))
    all_text_file_word_data['nb_of_terms'].append(len(word_data))
    all_text_file_word_data['word_data'].append(' '.join(word_data))
    all_text_file_word_data['full_word_data'].append(' '.join(full_word_data))

all_text_df = pd.DataFrame(all_text_file_word_data)
print(all_text_df.head())

# doc_term の出力
doc_term_file = f'{args.target}/{DOC_TERM_FILE}'
with open(doc_term_file, 'wt') as fout:
    logging.info(f'Create {doc_term_file}')
    for row in all_text_df.itertuples():
        fout.write(f'{row.nb_of_terms} {row.word_data}\n')
        logging.debug(f'{row.text_file}: {row.nb_of_terms} {row.word_data}')

############################################################
# LDA
############################################################

# LDA の推論
# - inf-gamma.dat と inf-lhood.dat を出力する
command = f'(cd {args.target}; {LDA} inf {args.settings} final {DOC_TERM_FILE} inf)'
ret = subprocess.getoutput(command)
logging.info(f'inference: {command} ({ret})')

# - inf-gamma.dat の読み込み
gamma_df = read_inf_gamma_file(args.target)
all_df = pd.concat([all_text_df, gamma_df], axis=1).sort_values(by='text_file')

# 出力
# output_file = os.path.join(args.target, datetime.now().strftime('result_%Y-%m-%d_%H%M.csv'))
output_file = f'{args.target}/result.csv'
# word_data 列は出力しない
all_df.drop(columns=['word_data'], inplace=True)
all_df.to_csv(output_file, index=False)

logging.info(f'Result is written to {output_file}.')
print(f'The result is written to {output_file}.', file=sys.stderr)
