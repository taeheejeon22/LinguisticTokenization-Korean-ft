# 감성 분석
# https://wikidocs.net/44249 참고

import argparse
import copy
import datetime
import json
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, '.')

import tokenization.tokenizers_v3_1 as Tokenizers
from functions import load_ft_model

from functools import partial

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard



## 1-1. 네이버 영화 리뷰 데이터에 대한 이해와 전처리
def load_and_preprocess():
    # 1) 데이터 로드하기
    train_data = pd.read_table('./for_test/NSMC/ratings_train.txt')
    test_data = pd.read_table('./for_test/NSMC/ratings_test.txt')

    # 2) 데이터 정제하기
    # for train data
    train_data['document'].nunique(), train_data['label'].nunique()
    train_data.drop_duplicates(subset=['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
    print('총 샘플의 수 :',len(train_data))

    print(train_data.isnull().values.any())
    print(train_data.isnull().sum())
    train_data.loc[train_data.document.isnull()]
    train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
    print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인

    print(train_data.isnull().sum())

    train_data = train_data.dropna(how = 'any')
    print(len(train_data))

    # for test data
    test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
    test_data = test_data.dropna(how='any') # Null 값 제거
    print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    return train_data, test_data


## 1-2. 토큰화
def tokenize_data(tokenizer: partial, dataframe: pd.core.frame.DataFrame):
    X_data = [" ".join(tokenizer(sent)) for sent in dataframe["document"]]
    return X_data


## 1-3. 정수 인코딩
def integer_encoding(X_train, X_test, train_data, test_data, min_count):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    threshold = min_count
    total_cnt = len(tokenizer.word_index) # 단어(실제로는 어절)의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print('단어 집합(vocabulary)의 크기 :',total_cnt)
    print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
    print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

    # 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
    # 0번 패딩 토큰을 고려하여 + 1
    vocab_size = total_cnt - rare_cnt + 1
    print('단어 집합의 크기 :',vocab_size)

    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])

    return X_train, X_test, y_train, y_test, tokenizer, vocab_size


## 1-4. 빈 샘플(empty samples) 제거
def remove_empty_samples(X_train, y_train):
    drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

    # 빈 샘플들을 제거
    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)
    print(len(X_train))
    print(len(y_train))

    return X_train, y_train


## 1-5. 패딩
def padding(X_train, X_test, max_len):
    print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
    print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))


    def below_threshold_len(max_len, nested_list):
      cnt = 0
      for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1
      print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

    below_threshold_len(max_len, X_train)

    X_train = pad_sequences(X_train, maxlen = max_len)
    X_test = pad_sequences(X_test, maxlen = max_len)

    return X_train, X_test, max_len


## 1-6. train set, dev set 나누기
def split_train_set(X_train, y_train):
    dev_indices = random.sample(range(len(X_train)), k=len(X_train)//10 )   # train set의 10%를 dev set으로 활용
    train_indices = [i for i in range(len(X_train)) if i not in dev_indices]

    X_train_final = X_train[train_indices]
    y_train_final = y_train[train_indices]

    X_dev = X_train[dev_indices]
    y_dev = y_train[dev_indices]

    print(len(X_train_final), len(y_train_final))
    print(len(X_dev), len(y_dev))

    return X_train_final, X_dev, y_train_final, y_dev


## 1-7. tensorflow pipeline 최적화
def get_tf_dataset(targets, labels, batch_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    BUFFER_SIZE = len(labels)
    dataset = tf.data.Dataset.from_tensor_slices((targets, labels))
    dataset = dataset.shuffle(BUFFER_SIZE, seed=seed).batch(batch_size=batch_size, drop_remainder=True)
    print(dataset)

    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    print(dataset)

    return dataset



# 2. LSTM으로 네이버 영화 리뷰 감성 분류하기
## 2-1. 임베딩 행렬 생성
def get_embedding_matrix(model, tokenizer, vocab_size):
    embed_size = model.get_dimension()  # 임베딩 차원

    embedding_matrix = np.zeros((vocab_size, embed_size))
    print("embedding_matrix shape:", embedding_matrix.shape)

    missed_token_indices = list()
    token2idx = list(tokenizer.word_index.items())[:vocab_size]

    for ix in range(len(token2idx)-1):
        token2idx_token = token2idx[ix][0]  # 토큰 (예: '하')
        token2idx_idx = token2idx[ix][1]  # 토큰 인코딩 인덱스 (예: 2)

        a_token = copy.deepcopy(token2idx_token)

        try:
            embedding_vector = model.get_word_vector(a_token)  # 로드한 fastText 벡터에서 헤당 토큰의 벡터 가져오기
            embedding_matrix[token2idx_idx] = embedding_vector  # 가져온 토큰 벡터를 임베딩용 빈 행렬에 넣기
        except KeyError:
            missed_token_indices.append(ix)
            embedding_matrix[token2idx_idx] = np.random.uniform(-1, 1, embed_size)

    return embedding_matrix, embed_size


## 2-2. 모델 정의
def get_LSTM_model(hyper_parameters, max_len, vocab_size, embed_size, embedding_matrix):
    use_pretrained_embedding = hyper_parameters["use_pretrained_embedding"]
    trainable = hyper_parameters["trainable"]
    lstm_units = hyper_parameters["lstm_units"]
    mask_zero = hyper_parameters["mask_zero"]
    learning_rate = hyper_parameters["learning_rate"]

    input = Input(shape=(max_len,), dtype="int32")

    init = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed)

    if use_pretrained_embedding == False:
        embeddings = Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=max_len, mask_zero=mask_zero, embeddings_initializer=init, trainable=trainable, name="embedding_layer1")(input)

    elif use_pretrained_embedding == True:
        embeddings = Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=max_len, mask_zero=mask_zero, weights=[embedding_matrix], trainable=trainable, name="embedding_layer1")(input)

    lstm0 = LSTM(units=lstm_units, dropout=0.25, return_sequences=True)(embeddings)
    lstm1 = LSTM(units=lstm_units, dropout=0.25, return_sequences=False)(lstm0)

    output = Dense(1, activation="sigmoid")(lstm1)

    model = Model(inputs=[input], outputs=output)

    Adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    return model



def main(test_setting, use_multi_gpus, hyper_parameters):
    model_sub_paths = sorted(os.listdir(ft_models_path))

    for ix in range(len(model_sub_paths)):
        ft_model_path = os.path.join(ft_models_path, model_sub_paths[ix])
        print(f"\n##################################################\n"
              f"model_sub_paths_ix: {ix}/{len(model_sub_paths)-1} \n"
              f"model_sub_paths {ft_model_path}\n"
              f"##################################################\n\n")

        ft_model_num = ft_model_path.split("/")[-1]

        print("test_setting", test_setting, sep="\n")

        # 테스트 결과 기록용
        test_results = test_setting.iloc[:,:6]

        # 테스트
        for jx in range(len(test_setting)):
            print(f"\n##################################################\n"
                  f"model_sub_paths_ix: {ix}/{len(model_sub_paths)-1} \n"
                  f"\niteration: {jx}\n"
                  f"##################################################\n\n")

            gpus = tf.config.list_logical_devices('GPU')
            mirrored_strategy = tf.distribute.MirroredStrategy(devices=[gpu.name for gpu in gpus])

            token_type = test_setting.loc[jx, "token_type"]

            tokenizer_type = test_setting.loc[jx, "tokenizer_type"]
            composition_type = test_setting.loc[jx, "composition_type"]
            dummy_letter = test_setting.loc[jx, "dummy_letter"] # composed / decomposed (nfd) / decomposed (old) 구별 위함

            min_count = test_setting.loc[jx, "nsmc_min_count"]
            max_len = test_setting.loc[jx, "nsmc_max_len"]

            hyper_parameters["nsmc_min_count"] = int(min_count)
            hyper_parameters["nsmc_max_len"] = int(max_len)

            print(f"token type: {token_type}\n"
                  f"tokenizer type: {tokenizer_type}\n"
                  f"composition type: {composition_type}\n"
                  f"dummy letter: {dummy_letter}\n"
                  )

            # 1-1. 데이터 로드 및 전처리
            train_data, test_data = load_and_preprocess()

            # 1-2. 토큰화
            # tokenizer 설정
            if dummy_letter == True:
                tok = Tokenizers.tokenizers(dummy_letter="⊸", nfd=False)
            elif dummy_letter == False:
                tok = Tokenizers.tokenizers(dummy_letter="", nfd=False)

            tokenize_fn = partial(tok.mecab_tokenizer, token_type=token_type, tokenizer_type=tokenizer_type, decomposition_type=composition_type)
            tokenization_example = tokenize_fn("훌륭한 사망 플래그의 예시이다.")
            print(f"tokenization example: {tokenization_example}")
            print(f"len_{tokenization_example[0]}: {len(tokenization_example[0])}\n"
                  f"len_{tokenization_example[1]}: {len(tokenization_example[1])}\n"
                  f"len_{tokenization_example[2]}: {len(tokenization_example[2])}\n")

            X_train = tokenize_data(tokenizer=tokenize_fn, dataframe=train_data)
            X_test = tokenize_data(tokenizer=tokenize_fn, dataframe=test_data)

            # 1-3. 정수 인코딩
            # integer_encoding(X_train=X_train, X_test=X_test, train_data=train_data, test_data=test_data, min_count=3)
            X_train, X_test, y_train, y_test, tokenizer, vocab_size = integer_encoding(X_train=X_train, X_test=X_test, train_data=train_data, test_data=test_data, min_count=min_count)

            # 1-4. 빈 샘플(empty samples) 제거
            X_train, y_train = remove_empty_samples(X_train=X_train, y_train=y_train)

            # 1-5. 패딩
            padding(X_train=X_train, X_test=X_test, max_len=25) # 길이 확인용
            X_train, X_test, max_len = padding(X_train=X_train, X_test=X_test, max_len=max_len)

            # 1-6. train set, dev set 나누기
            X_train, X_dev, y_train, y_dev = split_train_set(X_train=X_train, y_train=y_train)

            # 1-7. tensorflow pipeline 최적화
            train_data_set = get_tf_dataset(targets=X_train, labels=y_train, batch_size=hyper_parameters["batch_size"])
            dev_data_set = get_tf_dataset(targets=X_dev, labels=y_dev, batch_size=hyper_parameters["batch_size"])


            # 2-0. fastText 모델 로드
            print("load models")
            model = load_ft_model(ft_model_path=ft_model_path, token_type=token_type, tokenizer_type=tokenizer_type, composition_type=composition_type, dummy_letter=dummy_letter)

            # 2-1. 임베딩 행렬 생성
            embedding_matrix, embed_size = get_embedding_matrix(model=model, tokenizer=tokenizer, vocab_size=vocab_size)

            # 2-2. LSTM 모델 생성
            if use_multi_gpus == False:
                model = get_LSTM_model(hyper_parameters, max_len=max_len, vocab_size=vocab_size, embed_size=embed_size, embedding_matrix=embedding_matrix)
            elif use_multi_gpus == True:
                with mirrored_strategy.scope():
                    model = get_LSTM_model(hyper_parameters, max_len=max_len, vocab_size=vocab_size, embed_size=embed_size, embedding_matrix=embedding_matrix)

            # 2-3. 모델 학습
            es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)
            log_path = f"./log/NSMC/{ft_model_num}/{token_type}_{tokenizer_type}_{composition_type}_{'dummy_F' if dummy_letter == False else 'dummy_T'}"

            MC = ModelCheckpoint(f"{log_path}/best_model.h5", monitor="val_acc", mode="max", verbose=1, save_best_only=True)

            tb = TensorBoard(log_dir=f"{log_path}")

            history = model.fit(train_data_set, epochs=hyper_parameters["epochs"], callbacks=[es, MC, tb], validation_data=dev_data_set)

            # 테스트
            loaded_model = load_model(f"{log_path}/best_model.h5")

            loss, accuracy = loaded_model.evaluate(X_test, y_test)
            print("\n 테스트 정확도: %.4f" % (accuracy))

            test_results.loc[jx, "Accuracy"] = accuracy

            del model
            tf.keras.backend.clear_session()

        date = datetime.datetime.today().strftime('%Y-%m-%d')

        # 결과 저장
        result_file_path = f"./test_results/NSMC/NSMC_results_{ft_model_num}_{date}"
        os.makedirs("./test_results/NSMC/", exist_ok=True)
        test_results.to_csv(result_file_path + ".csv", index=False)

        # 하이퍼 파라미터 정보 저장
        with open(result_file_path + ".json", "w") as f:
            json.dump(hyper_parameters, f, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ft_models_path", type=str, default="")
    parser.add_argument("--test_settings", type=str, default="")

    parser.add_argument("--use_pretrained_embedding", action="store_true")
    parser.add_argument("--trainable", action="store_true")
    parser.add_argument("--mask_zero", action="store_true")

    parser.add_argument("--lstm_units", type=int, default=128)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=15)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_multi_gpus", action="store_true")

    args = vars(parser.parse_args())

    test_settings_path = args["test_settings"]

    print(args)


    # seed 설정
    seed = args["seed"]

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


    # test setting 파일 로드
    test_setting = pd.read_csv(test_settings_path)

    # fastText 모델 경로 설정
    ft_models_path = args["ft_models_path"]


    # main
    main(test_setting=test_setting, use_multi_gpus=args["use_multi_gpus"], hyper_parameters=args)
