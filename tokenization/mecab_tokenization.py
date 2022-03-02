# Mecab-ko 이용해 토큰화한 코퍼스 생성

import argparse
import json
import os
import re
import time

from functools import partial
from multiprocessing import Pool
from shutil import copyfile
from typing import List


import sys
sys.path.insert(0, '.')

import tokenization.tokenizers_v3_1 as Tokenizers


def tokenize(text: str, token_type: str, tokenizer_type: str, decomposition_type: str,  space_symbol: str = "▃", dummy_letter: str = "", flatten: bool = True, lexical_grammatical: bool = False) -> List[str]:
    text = text.strip()

    tokenized = tok.mecab_tokenizer(text, token_type=token_type, tokenizer_type=tokenizer_type, decomposition_type=decomposition_type, flatten=flatten, lexical_grammatical=lexical_grammatical)

    return tokenized



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_path", type=str)

    parser.add_argument("--token_type", type=str, default="")   # eojeol / morpheme # v2에서 추가
    parser.add_argument("--tokenizer_type", type=str, default="mecab_fixed")  # none / mecab_orig / mecab_fixed
    parser.add_argument("--decomposition_type", type=str, default="composed")   # "composed", "decomposed_pure", "decomposed_morphological"
    parser.add_argument("--dummy_letter", type=str, default="")  # 초성/중성/종성 자리 채우기용 더미 문자. default는 없음(""). # "⊸"  # chr(8888)

    parser.add_argument("--threads", type=int, default=12)

        # 추가 옵션 (fastText 실험에서는 필요 없음)
    parser.add_argument("--space_symbol", type=str, default="")  # "▃" chr(9603)
    parser.add_argument("--grammatical_symbol", type=list, default=["", ""])  # ["⫸", "⭧"] # chr(11000) # chr(11111)
    parser.add_argument("--lexical_grammatical", action="store_true", default=False)  # lexical_grammatical 분해할지. # 육식동물 / 에서 / 는
    parser.add_argument("--nfd", action="store_true", default=False)  # NFD 사용해서 자소 분해할지

    args = vars(parser.parse_args())
    print(args)


    # 출력 디렉토리 생성
    if args["dummy_letter"] == "":
        with_dummy_letter = "dummy_F"
    else:
        with_dummy_letter = "dummy_T"

    OUTPUT_DIR = f"./corpus/tokenized/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    INPUT_CORPUS = args["corpus_path"]
    corpus = "_".join(INPUT_CORPUS.split("/")[-1].split("_")[:2])   # corpus name


    p_endswith_num = re.compile("\d+$") # split 커맨드로 나눈 후 생기는 번호 검색용

    if p_endswith_num.search(INPUT_CORPUS): # 숫자로 끝나면. 즉 split 커맨드로 나뉜 파일이라면
        part_num = p_endswith_num.search(INPUT_CORPUS).group()

        if args["lexical_grammatical"] == False:
            file_name = "_".join([corpus, args["token_type"], args["tokenizer_type"], args["decomposition_type"], with_dummy_letter, part_num]) + ".txt"
        elif args["lexical_grammatical"] == True:  # lexical_grammtical 사용하면 morpheme 대신 LG로 명명
            file_name = "_".join([corpus, "LG", args["tokenizer_type"], args["decomposition_type"], with_dummy_letter, part_num]) + ".txt"

    else:
        if args["lexical_grammatical"] == False:
            file_name = "_".join([corpus, args["token_type"], args["tokenizer_type"], args["decomposition_type"], with_dummy_letter]) + ".txt"
        elif args["lexical_grammatical"] == True:  # lexical_grammtical 사용하면 morpheme 대신 LG로 명명
            file_name = "_".join([corpus, "LG", args["tokenizer_type"], args["decomposition_type"], with_dummy_letter]) + ".txt"


    # grammatical symbol
    symbol_josa = args["grammatical_symbol"][0]
    symbol_eomi = args["grammatical_symbol"][1]

    tok = Tokenizers.tokenizers(dummy_letter=args["dummy_letter"], space_symbol=args["space_symbol"], grammatical_symbol=[symbol_josa, symbol_eomi], nfd=args["nfd"])

    tokenize_fn = partial(tokenize, token_type=args["token_type"], tokenizer_type=args["tokenizer_type"], decomposition_type=args["decomposition_type"], lexical_grammatical=args["lexical_grammatical"])

    example = tokenize_fn("훌륭한 사망 플래그의 예시이다")
    print(f"tokenization example: {example}\n")


    # tokenization
    str_tokenizer_type = args["tokenizer_type"]
    str_decomposition_type = args["decomposition_type"]
    str_lexical_grammatical = str(args["lexical_grammatical"])
    print(f"corpus: {INPUT_CORPUS}\n"
          f"tokenizer_type: {str_tokenizer_type}\n"
          f"decomposition_type: {str_decomposition_type}\n"
          f"lexical_gramatical: {str_lexical_grammatical}\n")


    start_time = time.time()
    print(f"start tokenization...\n")

    if (args["tokenizer_type"] == "none") and (args["decomposition_type"] == "composed"):    # 형태소 분석하지 않고 원문 그대로 이용
        pass

    else:   # 형태소 분석할 경우
        with open(INPUT_CORPUS, "r", encoding="utf-8") as f:
            with Pool(args["threads"]) as p:
                tokenized = p.map(tokenize_fn, f)


    # 출력 경로 설정
    if args["lexical_grammatical"] == False:
        OUTPUT_DIR_sub = OUTPUT_DIR + "_".join([args["token_type"], args["tokenizer_type"] ]) + "/" + "_".join([args["decomposition_type"], with_dummy_letter])
    elif args["lexical_grammatical"] == True:  # lexical_grammtical 사용하면 morpheme 대신 LG로 명명
        OUTPUT_DIR_sub = OUTPUT_DIR + "_".join(["LG", args["tokenizer_type"] ]) + "/" + "_".join([args["decomposition_type"], with_dummy_letter])


    os.makedirs(OUTPUT_DIR_sub, exist_ok=True)


    # 저장
    OUTPUT_PATH = os.path.join(OUTPUT_DIR_sub, os.path.basename(file_name))

    if (args["tokenizer_type"] == "none") and (args["decomposition_type"] == "composed"):    # 형태소 분석하지 않고 원문 그대로 이용
        copyfile(INPUT_CORPUS, OUTPUT_PATH)

    else:  # 형태소 분석할 경우
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for tokens in tokenized:
                f.write(" ".join(tokens) + "\n")



    # tokenization config
    print("write tokenization config file...\n")
    output_config_path = os.path.join(OUTPUT_DIR_sub, "tok.json")
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=4)


    print(f"saved in: {os.path.join(OUTPUT_DIR_sub, os.path.basename(file_name))}\n")
    print(f"done.\n")


    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"complete tokenization for all files. (elapsed time: {elapsed_time})\n")
