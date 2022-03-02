# 유사도, 유추 테스트
# 테스트 세트: https://github.com/SungjoonPark/KoreanWordVectors

import argparse
import datetime
import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '')

from functions import load_ft_model
from functools import partial
from scipy.stats import pearsonr, spearmanr

import tokenization.tokenizers_v3_1 as Tokenizers



# 유추 테스트 세트 분리: Case, Voice, Tense, ...
def split_testset(file_path: str):
    with open(file_path, "r") as f:
        testset_all = f.readlines()
        testset_all = [line[:-1] for line in testset_all]

    start_idxs = [idx for idx, cont in enumerate(testset_all) if ": " in cont]
    end_idxs = [idx-1 for idx in start_idxs[1:]] + [len(testset_all)-1]

    testset_split = list()

    for ix in range(len(start_idxs)):
        testset_split.append( testset_all[start_idxs[ix]:end_idxs[ix]] )

    return testset_split


# 단어를 형태소 분석 후 형태소 벡터들의 합으로 단어 벡터 구하기
def get_word_vector(word: str, tokenizer: partial, use_gensim: bool, model):
    def get_token_vector(token: str):
        if use_gensim == True:
            token_vector = model.wv.get_vector(token, norm=False)
        elif use_gensim == False:
            token_vector = model.get_word_vector(token)
        return token_vector

    morphemes = tokenizer(word)
    word_vector = sum([get_token_vector(token=morpheme) for morpheme in morphemes])
    return word_vector


# OOV 여부 카운트
def check_vocabulary(word: str, tokenizer: partial, use_gensim: bool, model):
    tokenized = tokenizer(word)

    oov_cnt = 0

    if use_gensim == True:
        for token in tokenized:
            if not (token in model.wv.key_to_index):    # 토큰이 하나라도 vocabulary에 없으면 oov_cnt 추가
                oov_cnt += 1
                break
    elif use_gensim == False:
        for token in tokenized:
            if not (token in model.words):
                oov_cnt += 1
                break

    return oov_cnt


### 유사도 테스트
# 코사인 유사도 계산
def cos_sim(A, B):
    return np.inner(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


# 단어 벡터 쌍의 코사인 유사도 계산
def get_similarity(tokenizer: partial, word1, word2, model):
    # oov word 하나당 count 1
    oov_count = sum([check_vocabulary(word=word, tokenizer=tokenizer, use_gensim=use_gensim, model=model) for word in [word1, word2]])

    # get word vectors
    w1_vec = get_word_vector(word=word1, tokenizer=tokenizer, use_gensim=use_gensim, model=model)
    w2_vec = get_word_vector(word=word2, tokenizer=tokenizer, use_gensim=use_gensim, model=model)

    pred_sim = cos_sim(w1_vec, w2_vec)

    return pred_sim, oov_count


# 유사도 테스트
def similarity_test(tokenizer: partial, file_path: str, model, token_type: str, composition_type: str,):

    total = 0
    pred_sim_list = list()
    true_sim_list = list()

    missed = 0
    missed_lst = list()

    oov_count_sum = 0
    oov_list = list()

    kor_ws353 = pd.read_csv(file_path, header=None)

    for ix in range(len(kor_ws353)):
        total += 1

        word1 = kor_ws353.iloc[ix,0]
        word2 = kor_ws353.iloc[ix,1]
        true_sim = kor_ws353.iloc[ix,2]

        pred_sim, oov_count = get_similarity(tokenizer=tokenizer, word1=word1, word2=word2, model=model,)

        if np.isnan(pred_sim):  # 유사도 계산 안 되면 (OOV 계산 위한 n-gram 벡터가 아예 없는 경우)
            missed += 1
            missed_lst.append([word1, word2])

        if oov_count >= 1:  # oov 포함된 세트라면
            oov_list.append((word1, word2))

        # OOV 있어도 모두 계산에 포함
        oov_count_sum += oov_count
        pred_sim_list.append(pred_sim)
        true_sim_list.append(true_sim)


    pearsonr_r, _ = pearsonr(true_sim_list, pred_sim_list)
    spearman_r, _ = spearmanr(true_sim_list, pred_sim_list)

    print(f"total: {total}")
    print(f"missed: {missed}")
    print(f"oov count: {oov_count_sum}")
    print(f"pearson r: {pearsonr_r}")
    print(f"spearman r: {spearman_r}")

    return pearsonr_r, spearman_r, pred_sim_list, oov_count_sum, oov_list



### 유추 테스트
# # 3CosAdd (Park et al., 2018): 논문의 설명과 식이 다름. 이 식 사용하면 3CosAdd 제대로 구해지지 않음.
# def get_3cosadd_dist(a, b, c, d):
#     vec_a = a + b - c
#     vec_b = d
#
#     dist = 1 - cos_sim(vec_a, vec_b)
#
#     return dist
# 3CosAdd (Levy & Goldberg, 2014): 3CosAdd 처음 제안한 논문. 이 식에 기반하여 계산해야 제대로 된 3CosAdd 얻을 수 있음.
def get_3cosadd_dist(a, b, c, d):
    vec_a = c + (b - a) # 유추된 벡터
    vec_b = d   # 정답 벡터

    dist = 1 - cos_sim(vec_a, vec_b)

    return dist


def get_3cosmul_dist(a, b, c, d, epsilon=0.001):
    cosmul = cos_sim(d, b)*cos_sim(d, c) / (cos_sim(d, a) + epsilon)

    dist = 1 - cosmul

    return dist


'''
https://github.com/dongjun-Lee/kor2vec/blob/master/test/analogy_test.py 참고
'''
def analogy_test(tokenizer: partial, testset: list, model, token_type: str, composition_type: str, ):

    total = 0
    cos_add_dist_lst = list()
    cos_mul_dist_lst = list()

    oov_count_sum = 0
    oov_list = list()


    for ix in range(len(testset)):
        if ": " in testset[ix]: # ': sem1_capital-conturies' 등의 행 스킵
            continue

        total += 1

        word1, word2, word3, word4 = testset[ix].strip().split(" ")  # ['대한민국', '서울', '일본', '도쿄']

        # get word vectors (v2.1)
        vec1 = get_word_vector(word=word1, tokenizer=tokenizer, use_gensim=use_gensim, model=model)
        vec2 = get_word_vector(word=word2, tokenizer=tokenizer, use_gensim=use_gensim, model=model)
        vec3 = get_word_vector(word=word3, tokenizer=tokenizer, use_gensim=use_gensim, model=model)
        vec4 = get_word_vector(word=word4, tokenizer=tokenizer, use_gensim=use_gensim, model=model)


        # OOV word 하나당 카운트 1
        oov_count = sum([check_vocabulary(word=word, tokenizer=tokenizer, use_gensim=use_gensim, model=model) for word in [word1, word2, word3, word4]])
        oov_count_sum += oov_count

        if oov_count >= 1:  # oov 포함된 세트라면
            oov_list.append((word1, word2))

        cos_add_dist = get_3cosadd_dist(vec1, vec2, vec3, vec4)
        cos_mul_dist = get_3cosmul_dist(vec1, vec2, vec3, vec4, epsilon=0.001)

        cos_add_dist_lst.append(cos_add_dist)
        cos_mul_dist_lst.append(cos_mul_dist)


    mean_cos_add_dist = np.mean(cos_add_dist_lst)
    mean_cos_mul_dist = np.mean(cos_mul_dist_lst)

    print(f"total: {total}")
    print(f"oov count: {oov_count_sum}")
    print(f"mean_cos_add_dist: {mean_cos_add_dist}")
    print(f"mean_cos_mul_dist: {mean_cos_mul_dist}")

    return mean_cos_add_dist, mean_cos_mul_dist, oov_count_sum, oov_list



def main(test_setting: pd.core.frame.DataFrame, similarity_test_file_path: str, analogy_test_semantic_file_path: str, analogy_test_syntactic_file_path: str, use_gensim: bool):
    model_sub_paths = sorted(os.listdir(ft_models_path))

    for ix in range(len(model_sub_paths)):
        ft_model_path = os.path.join(ft_models_path, model_sub_paths[ix])
        print(f"\n##################################################\n"
              f"model_sub_paths_ix: {ix}/{len(model_sub_paths) - 1} \n"
              f"model_sub_paths {ft_model_path}\n"
              f"##################################################\n\n")

        ft_model_num = ft_model_path.split("/")[-1]

        print("test_setting", test_setting, sep="\n")

        # analogy 전체
        with open(analogy_test_semantic_file_path, "r") as f:
            analogy_test_whole_semantic = [x[:-1] for x in f.readlines() if (x != "\n") and (not x.startswith("#") and not x.startswith(":"))]
        with open(analogy_test_syntactic_file_path, "r") as f:
            analogy_test_whole_syntactic = [x[:-1] for x in f.readlines() if (x != "\n") and (not x.startswith("#") and not x.startswith(":"))]

        # analogy 세부 항목별
        analogy_test_semantic = split_testset(file_path=analogy_test_semantic_file_path)
        analogy_test_syntactic = split_testset(file_path=analogy_test_syntactic_file_path)

        # analogy 데이터 통합 (항목별 + 전체)
        analogy_test_semantic.append(analogy_test_whole_semantic)
        analogy_test_syntactic.append(analogy_test_whole_syntactic)


        # 테스트 결과 기록용
        test_results = test_setting.iloc[:,:6]
        analogy_test_semantic_results = np.zeros(shape=(len(analogy_test_semantic), 3)) # len(analogy_test_semantic): 모든 세부 테스트(5) + 전체 테스트(1)     # 3: cosADD, codMul, oov
        analogy_test_syntactic_results = np.zeros(shape=(len(analogy_test_syntactic), 3))


        # 테스트
        for jx in range(len(test_setting)):
            print(f"\n##################################################\n"
                  f"model_sub_paths_ix: {ix}/{len(model_sub_paths) - 1} \n"
                  f"\niteration: {jx}\n"
                  f"##################################################\n\n")

            token_type = test_setting.loc[jx, "token_type"]

            tokenizer_type = test_setting.loc[jx, "tokenizer_type"]
            composition_type = test_setting.loc[jx, "composition_type"]
            dummy_letter = test_setting.loc[jx, "dummy_letter"] # composed / decomposed (nfd) / decomposed (old) 구별 위함

            # fasttext 모델 로드
            print("load models\n")
            model = load_ft_model(ft_model_path=ft_model_path, token_type=token_type, tokenizer_type=tokenizer_type, composition_type=composition_type, dummy_letter=dummy_letter, use_gensim=use_gensim)

            # tokenizer 설정
                # 인스턴스 생성
            if dummy_letter == True:
                tok = Tokenizers.tokenizers(dummy_letter="⊸", nfd=False)
            elif dummy_letter == False:
                tok = Tokenizers.tokenizers(dummy_letter="", nfd=False)

            tokenize_fn = partial(tok.mecab_tokenizer, token_type=token_type, tokenizer_type=tokenizer_type, decomposition_type=composition_type)
            tokenization_example = tokenize_fn("훌륭한 사망 플래그의 예시이다.")
            print(f"tokenization example: {tokenization_example}")


            # 유사도 테스트
            pearson_r, spearman_r, preds, oov_count_sim, oov_list_sim = similarity_test(tokenizer=tokenize_fn, file_path=similarity_test_file_path, model=model, token_type=token_type, composition_type=composition_type)

            pearson_r = round(pearson_r, 3)
            spearman_r = round(spearman_r, 3)


            # 유추 테스트: semantic
            for kx in range(len(analogy_test_semantic)):
                CosAdd_dist, CosMul_dist, oov_count_sem, oov_list_sem = analogy_test(tokenizer=tokenize_fn, testset=analogy_test_semantic[kx], model=model, token_type=token_type, composition_type=composition_type)
                analogy_test_semantic_results[kx] = round(CosAdd_dist, 3), round(CosMul_dist, 3), oov_count_sem


            # 유추 테스트: syntactic
            for lx in range(len(analogy_test_syntactic)):
                CosAdd_dist, CosMul_dist, oov_count_syn, oov_list_syn = analogy_test(tokenizer=tokenize_fn, testset=analogy_test_syntactic[lx], model=model, token_type=token_type, composition_type=composition_type)
                analogy_test_syntactic_results[lx] = round(CosAdd_dist, 3), round(CosMul_dist, 3), oov_count_syn

            del model  # for saving memory


            test_results.loc[jx, "Similarity_Pearson's r"] = pearson_r
            test_results.loc[jx, "Similarity_Spearman's \u03C1"] = spearman_r
            test_results.loc[jx, "Similarity_OOV Count"] = oov_count_sim


            sub_types_semantic = ["Capt", "Gend", "Name", "Lang", "Misc", "Total"]
            for mx in range(len(analogy_test_semantic_results)):
                sub_type = sub_types_semantic[mx]

                test_results.loc[jx, f"Analogy Semantic {sub_type}_3CosAdd Distance"] = analogy_test_semantic_results[mx][0] # 3CosAdd dist.
                test_results.loc[jx, f"Analogy Semantic {sub_type}_OOV Count"] = analogy_test_semantic_results[mx][2] # 3CosAdd dist.


            sub_types_syntactic = ["Case", "Tense", "Voice", "Form", "Honr", "Total"]
            for mx in range(len(analogy_test_syntactic_results)):
                sub_type = sub_types_syntactic[mx]

                test_results.loc[jx, f"Analogy Syntactic {sub_type}_3CosAdd Distance"] = analogy_test_syntactic_results[mx][0]
                test_results.loc[jx, f"Analogy Syntactic {sub_type}_OOV Count"] = analogy_test_syntactic_results[mx][2] # 3CosAdd dist.


        # 결과 저장
        date = datetime.datetime.today().strftime('%Y-%m-%d')

        result_file_path = f"./test_results/sim_anal/sim_anal_results_{ft_model_num}_{date}.csv"
        test_results.to_csv(result_file_path, index=False)

        print(f"result saved in: {result_file_path}")
        print("complete!")



if __name__ == "__main__":
    # 테스트 파일 경로
    similarity_test_file_path = "./for_test/WS353_korean_SungjoonPark.csv"
    analogy_test_semantic_file_path = "./for_test/kor_analogy_semantic_SungjoonPark_all.txt"
    # analogy_test_syntactic_file_path = "./for_test/kor_analogy_syntactic_SungjoonPark_all.txt"    # Park et al. (2018) 데이터 세트의 원본 기반 버전
    analogy_test_syntactic_file_path = "./for_test/kor_analogy_syntactic_SungjoonPark_all_v2.txt"   # Park et al. (2018) 데이터 세트 중 형태소 분석 토큰화에 유리한 것들 배제한 버전

    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_models_path", type=str, default="")
    parser.add_argument("--test_settings", type=str, default="")
    parser.add_argument("--use_gensim", action="store_true", default=False)
    # API를 gensim으로 할지 official로 할지. # 모델 학습을 뭐로 했는지랑은 상관없음. # 서로 차이가 없는 게 정상.
    # use_gensim = True   # use gensim (https://github.com/RaRe-Technologies/gensim)
    # use_gensim = False  # fastText python implementation (https://github.com/facebookresearch/fastText) # faster

    args = vars(parser.parse_args())

    test_settings_path = args["test_settings"]

    print(args)

    test_setting = pd.read_csv(test_settings_path)

    ft_models_path = args["ft_models_path"]
    use_gensim = args["use_gensim"]

    main(test_setting=test_setting, similarity_test_file_path=similarity_test_file_path, analogy_test_semantic_file_path=analogy_test_semantic_file_path, analogy_test_syntactic_file_path=analogy_test_syntactic_file_path, use_gensim=use_gensim)
