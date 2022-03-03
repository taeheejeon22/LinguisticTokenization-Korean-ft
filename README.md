# Requirements
- OS: Ubuntu (16.04 이상)
- Python: 3.8 이상
- 하드웨어
  - RAM
    * 최소: 64 GB (코퍼스 크기 문제로 인해 Out of Memory 문제 발생할 수 있음)
    * 권장: 128 GB 이상
  - CPU: 12 스레드 이상 CPU 이용 권장 (예: AMD Ryzen 5 3600X, Intel Core i5-12400)
  - GPU: CPU의 다중 작업 성능이 충분히 좋으면(대략 Intel Core i9-9900K 급 이상) 없어도 무방함. 사용한다면 GeForce 10 시리즈 이상 이용을 권장하며, 간단한 모델을 이용하므로 VRAM이 큰 하이엔드 모델이 아니어도 됨 


# **튜토리얼**
- 가상 환경 구축하는 것을 권장
- 아래의 커맨드가 작동하지 않으면 pip, python 대신 pip3, python3를 사용해 볼 것


## 0. 필요 라이브러리 설치
```bash
pip install -r requirments.txt
```
- 자신의 하드웨어, CUDA, Python 버전에 맞는 TensorFlow 버전 설치
  * https://www.tensorflow.org/install/source 하단의 표 참조

- Mecab-ko 설치는 다음 참조
  * https://konlpy.org/en/latest/install/
- 가상 환경에 Mecab-ko 설치하기
  * 터미널에서 가상 환경 활성화 후 https://konlpy.org/en/latest/install/ 참조하여 Mecab-ko 설치
  * 가상 환경 활성화된 상태에서 아래 커맨드 입력
  ```bash
  cd /tmp
  cd mecab-python-0.996/
  python setup.py install
  ```


## 1. 임베딩 학습용 코퍼스 만들기
### 1) 나무위키 코퍼스 덤프 파일(json) 다운로드 
https://mu-star.net/wikidb


### 2) json 파일을 문장별로 분리된 텍스트 파일로 변환
```bash
python preprocessing/namuwiki_preprocessing.py --corpus_path=../namuwiki_20210301.json --threads=12
```
- 12-16 스레드 사용 시 2-3 시간 정도 소요

#### 사용법
- corpus_path: json 파일 경로. 자신이 다운로드한 json 파일의 경로로 수정할 것
- output_path: 결과 파일 저장 경로. 설정하지 않으면 corpus_path와 동일한 곳에 저장됨 (파일명 끝부분이 _preprocessed.txt로 변경)
- threads: 사용할 스레드 수 (default: 12)


### 3) (optional) 코퍼스 전체에서 문장 랜덤 샘플링
```bash
python preprocessing/random_sampling.py --corpus_path=../namuwiki_20210301_preprocessed.txt --seed=22 --sampling_ratio=0.25
```

#### 사용법
- corpus_path: 1)에서 만들어진 텍스트 파일 경로. 1)에서 output_path 설정을 하지 않았으면 위 커맨드 그대로 사용하면 되고, 설정을 했으면 그 경로를 적으면 됨 
- seed: 랜덤 시드 (default = 22)
- sampling_ratio: 샘플링할 비율 (default = 0.25)
- output_path: 결과 파일 저장 경로. 설정하지 않으면 corpus_path와 동일한 폴더 내에 저장됨 (파일명 끝부분이 _sampled.txt로 변경)



## 2. 코퍼스 토큰화
```bash
python tokenization/mecab_tokenization.py --corpus_path=../namuwiki_20210301_preprocessed_sampled.txt --token_type=eojeol --tokenizer_type=none --decomposition_type=composed --threads=12
python tokenization/mecab_tokenization.py --corpus_path=../namuwiki_20210301_preprocessed_sampled.txt --token_type=eojeol --tokenizer_type=none --decomposition_type=decomposed_simple --dummy_letter=⊸ --threads=12

python tokenization/mecab_tokenization.py --corpus_path=../namuwiki_20210301_preprocessed_sampled.txt --token_type=morpheme --tokenizer_type=mecab_fixed --decomposition_type=composed --threads=12
python tokenization/mecab_tokenization.py --corpus_path=../namuwiki_20210301_preprocessed_sampled.txt --token_type=morpheme --tokenizer_type=mecab_fixed --decomposition_type=decomposed_simple --dummy_letter=⊸ --threads=12
python tokenization/mecab_tokenization.py --corpus_path=../namuwiki_20210301_preprocessed_sampled.txt --token_type=morpheme --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --dummy_letter=⊸ --threads=12
python tokenization/mecab_tokenization.py --corpus_path=../namuwiki_20210301_preprocessed_sampled.txt --token_type=morpheme --tokenizer_type=mecab_fixed --decomposition_type=decomposed_grammatical --dummy_letter=⊸ --threads=12
```
- KoNLPy의 Mecab-ko 모듈을 수정한 파일(./_mecab.py) 이용해 토큰화
- ./pretrain_corpus/tokenized 에 저장됨
- 12-16 스레드 사용 시 1-2 시간 소요

#### 사용법
- corpus_paht: 1.에서 만들어진 코퍼스 파일 경로. 1. 3)에서 만들어진 파일의 경로를 확인한 후 그에 맞게 수정해 사용할 것
- token_type: 토큰화 단위
  * eojeol: 어절
  * morpheme: 형태소
- tokenizer: 토크나이저 유형
  * none: 없음 (token_type이 eojeol일 시 사용)
  * mecab_fixed: 수정된 Mecab-ko
  * mecab_orig: 원본 Mecab-ko
- decomposition_type: 자소 분해 유형 
    * composed: 분해 없이 음절 유지
    * decomposed_simple: 단순 자소 분해
    * decomposed_lexical: 어휘 형태소만 자소 분해
    * decomposed_grammatical: 문법 형태소만 자소 분해
- dummy_letter: 초성, 중성, 종성 자리 채우기용 더미 문자. 설정하지 않으면 더미 문자 없음. (default = "")
- threads: 사용할 스레드 수 (default = 12)



## 2. fastText 학습
```bash
python scripts/train_fasttext.py --parameter_settings=./settings/ft_param_settings.csv --iteration=5 --threads=12
```
- 12-16 스레드 사용 + iteration 5 설정 시 1-2 일 소요

#### 사용법
- parameter_settings: fastText 세팅별 임베딩 학습 하이퍼 파라미터 기록한 파일 경로. 파일을 열어 하이퍼 파라미터 직접 설정 가능
- iteration: 1세팅당 임베딩 모델 몇 회 학습할지. 무작위 초기화 파라미터로 인한 우연적인 결과 막기 위함 (default = 5)
- threads: 사용할 스레드 수 (default = 12)
- corpus_path: 코퍼스 파일들이 저장되어 있는 경로 (default = "./corpus/tokenized/")



## 3. test
### 1) 유사도, 유추 테스트
```bash
python scripts/test_sim_syn.py --ft_models_path=./models --test_settings=./settings/test_settings.csv
```
- 모델 인스턴스 5개인 경우(2.에서 iteration 5로 설정), 30분 내외 소요

#### 사용법
- ft_models_path: 2.에서 학습한 모델들 저장된 경로
- test_settings: fastText 세팅별 테스트 하이퍼 파라미터 기록한 파일 경로
- use_gensim: gensim 라이브러리 사용 여부
  * --use_gensim: gensim 라이브러리 사용하여 fastText 모델 로드
  * 아무 입력 안 함: fastText 공식라이브러리 사용하여 fastText 모델 로드


### 2) 감성 분석 (NSMC)
```bash
python scripts/test_NSMC.py --ft_models_path=./models --test_settings=./settings/test_settings.csv --use_pretrained_embedding --mask_zero 
```
- 모델 인스턴스 5개인 경우(2.에서 iteration 5로 설정), CPU 사용 시 1-2일 소요, GPU 1개 사용 시 12시간 내외 소요

#### 사용법
- ft_models_path: 2.에서 학습한 모델들 저장된 경로
- test_setting_path: fastText 세팅별 테스트 하이퍼 파라미터 기록한 파일 경로
- use_pretrained_embedding: 2.에서 학습한 임베딩 모델 이용해서 임베딩 레이어 구성할지, 랜덤하게 초기화해서 임베딩 레이어 구성할지
  * --use_pretrained_embedding: 2.에서 학습한 임베딩 모델 이용
  * 아무 입력 안 함: 랜덤하게 초기화
- trainable: 임베딩 레이어의 파라미터들을 학습할지 여부
  * --trainable: 학습함
  * 아무 입력 안 함: 학습 안 함
- mask_zero: 0으로 패딩되는 노드들을 무시할지 여부
  * --mask_zero: 무시
  * 아무 입력 안 함: 무시 안 함
- batch_size: 배치 사이즈 (default = 64)
- learning_rate: 학습률 (default = 1e-4)
- epochs: 에포크 수 (default = 15)
- seed: 랜덤 시드 설정 (default = 42)
- use_multi_gpus: GPU를 2개 이상 사용할지 여부 
  * --use_multi_gpus: GPU를 2개 이상 사용
  * 아무 입력 안 함: GPU 1개 사용
  * 실질적인 속도 향상을 위해서는 GPU 수(N개)만큼 batch size를 N배 해 주어야 함. 설정한 batch를 GPU N개에서 나누어 처리하는 방식이므로, batch size가 그대로이면 속도 향상 없음. (예: 1 GPU 학습 시 batch size 128이면, 2 GPU 학습 시 batch size 256으로 설정)
  * GPU가 여러 개인 환경에서 1개만 쓰려면, 위 코드 앞에 CUDA_VISIBLE_DEVICES=0 과 같이 사용할 GPU의 인덱스를 명시해야 함. 하지 않으면 TensorFlow가 모든 GPU를 로드하므로, 실제로 학습을 하지 않는 GPU가 있더라도 이를 다른 용도로 쓰지 못함
  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/test_NSMC.py --ft_models_path=./models --test_settings=./settings/test_settings.csv --use_pretrained_embedding --mask_zero
  ```


### 3) 테스트 결과 평균
```bash
python scripts/average_results.py --results_path=./test_results/sim_anal
python scripts/average_results.py --results_path=./test_results/NSMC
```
- 모델 인스턴스들(기본 5개)의 학습 결과 평균
- test_results/average 경로에 저장됨


### 4) (optional) tensorboard에서 학습 결과 확인
```bash
tensorboard --logdir=log/NSMC/
```
#### 사용법
- logdir: 학습 로그 저장 경로 (코드 수정을 하지 않았다면 ./log/NSMC에 저장됨) 


#### Tensorboard on a remote server
[//]: # (https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server)
```bash
ssh -L 16006:127.0.0.1:6006 jth@14.39.30.242          # 서버의 6006 포트를 로컬의 16006 포트로 수신
tensorboard --logdir=log/NSMC/eojeol_none_composed_False
```

#### local
```bash
http://127.0.0.1:16006  # 로컬의 16006 포트에서 수신
```



