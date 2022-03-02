# 나무위키 전처리

## 실행 전에 나무위키 덤프 파일 받아 둘 것
# 덤프 파일
# repository: https://mu-star.net/wikidb
# version: 2021/03/01

# extractor
# https://github.com/jonghwanhyeon/namu-wiki-extractor

import argparse
import json
import re
import time

import kss

from multiprocessing import Pool
from namuwiki.extractor import extract_text


# 전처리
def preprocess(sent_lst: list):
    # 괄호 문자열("(xxx)") 삭제
    p_paren_str = re.compile("\(.+?\)")
    sent_lst = [re.sub(p_paren_str, "", sent) for sent in sent_lst] # 사람(인간)은 짐승(동물)이다 > 사람은 짐승이다

    # 타 언어 문자, 특수 기호 삭제
    p_exotic = re.compile(r"[^ㄱ-ㅎㅏ-ㅣ가-힣\x20-\x7F]*")
    sent_lst = [re.sub(p_exotic, "", sent) for sent in sent_lst]

    # 무의미한 공백 삭제
    p_multiple_spaces = re.compile("\s+")
    sent_lst = [re.sub(p_multiple_spaces, " ", sent) for sent in sent_lst]  # 무의미한 공백을 스페이스(" ")로 치환

    # 빈 라인 삭제
    sent_lst = [sent for sent in sent_lst if not re.search(r"^\s+$", sent)]
    sent_lst = [sent.strip() for sent in sent_lst if sent != ""]

    # 짧은 라인 없애기
    sent_lst = [sent for sent in sent_lst if len(sent.split(" ")) >= 3 ]   # 예: 퇴임 이후. 어린 시절.  생애 후반.

    return sent_lst


# 위키 문서를 1행 1문장의 문자열로 변환
def process_document(document) -> str:
    plain_text = extract_text(document['text'])

    # 1행 1문장이 되도록 문장 분리
    split_text0 = kss.split_sentences(plain_text)
    split_text1 = [text.splitlines() for text in split_text0]
    split_text2 = [text for text_list in split_text1 for text in text_list]

    preprocessed_text = preprocess(sent_lst=split_text2)

    concat_text = "\n".join(preprocessed_text) + "\n"

    return concat_text



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_path", type=str, default="")  # 나무위키 덤프 파일 경로
    parser.add_argument("--threads", type=int, default=12)  # 사용 스레드 수
    parser.add_argument("--output_path", type=str, default="")  # 출력 파일 경로

    args = vars(parser.parse_args())
    print(args)

    start_time = time.time()


    # 원본 코퍼스 파일 로드
    with open(args["corpus_path"], 'r', encoding='utf-8') as input_file:
        corpus = json.load(input_file)
        print("\nCorpus loaded!\n")


    # 전처리
    print("Preprocessing...")
    with Pool(args["threads"]) as p:
        documents = p.map(process_document, corpus)
        print("Preprocessing finished!\n")


    # 저장 경로
    if args["output_path"] == "":   # If there is no output_path
        output_path = args["corpus_path"].split(".json")[0] + "_preprocessed.txt"
    else:
        output_path = args["output_path"]


    # 저장
    with open(output_path, "w") as f:
        for ix in range(len(documents)):
            if documents[ix] != "\n":
                f.write(documents[ix])


    print(f"\nsaved in: {output_path}\n")
    print("Done!\n")


    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"(elapsed time: {elapsed_time})")

    # Intel Core i9-12900K (24 스레드 사용) 소요 시간
    # (elapsed time: 00:47:32)
