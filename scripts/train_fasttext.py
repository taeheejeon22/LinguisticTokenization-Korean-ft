# fastText 학습

import argparse
import os
import time

import fasttext
import pandas as pd


# function for model training and saving
def train_and_save(params: dict, file_path: str, token_type: str, tokenizer_type: str, composition_type: str, dummy_letter: str, save_dir: str, threads: int = 12):
    # hyper-parameters
    dim = params["dim"]
    ws = params["ws"]
    minCount = params["minCount"]
    minn = params["minn"]
    maxn = params["maxn"]
    lr = params["lr"]
    loss = params["loss"]

    # load a raw corpus
    with open(file_path, "r") as f:
        corpus = f.readlines()
        corpus = [line.split() for line in corpus]

    print(f"corpus_example: {corpus[0]}\n"
          f"{corpus[1]}")

    # train
    model = fasttext.train_unsupervised(file_path, thread=threads, dim=dim, ws=ws, minn=minn, maxn=maxn, loss=loss, lr=lr, model='skipgram')  # official

    print(model)

    # save
    save_path = save_dir + "/" + "_".join(
        [token_type, composition_type, dummy_letter, "dim", str(dim), "ws", str(ws), "minCount", str(minCount), "minn", str(minn), "maxn", str(maxn), "lr", str(lr)]
    )

    model.save_model(save_path) # official

    print(f"\nsaved in: {save_path}\n")


# get the full path of files
def listdir_fullpath(path):
    return sorted([os.path.join(path, f) for f in os.listdir(path)])


def main(iteration, threads, corpus_path, param_setting):

    for jx in range(iteration):

        for ix in range(len(param_setting)):

            start_time = time.time()

            print(f"\n\niteration: {jx}\n")
            print(f"setting: {ix}\n")

            token_type = param_setting.loc[ix, "token_type"]

            tokenizer_type = param_setting.loc[ix, "tokenizer_type"]

            composition_type = param_setting.loc[ix, "composition_type"]
            dummy_letter = "dummy_F" if param_setting.loc[ix, "dummy_letter"] == False else "dummy_T"

            sub_path0 = "_".join([token_type, tokenizer_type])
            sub_path1 = "_".join([composition_type, dummy_letter])

            corpus_path_lst = listdir_fullpath( "/".join([corpus_path, sub_path0, sub_path1]).replace("//", "/") )
            corpus_path_lst = [path for path in corpus_path_lst if path.endswith(".txt")]

            print(f"corpusl_path: {corpus_path_lst}\n")

            params = {
            "dim": param_setting.loc[ix, "dim"],
            "ws": param_setting.loc[ix, "ws"],
            "minCount": param_setting.loc[ix, "minCount"],
            "minn": param_setting.loc[ix, "minn"],
            "maxn": param_setting.loc[ix, "maxn"],
            "lr": param_setting.loc[ix, "lr"],
            "loss": param_setting.loc[ix, "loss"]
            }


            # 저장
            model_iteration = f"model_{jx}"
            # save_dir = "./models/" + "_".join([param_ver, str(jx)])  + "/" + "_".join([token_type, tokenizer_type, composition_type, dummy_letter, nfd])
            save_dir = f"./models/{model_iteration}/" + "_".join([token_type, tokenizer_type, composition_type, dummy_letter])
            os.makedirs(save_dir, exist_ok=True) # 리눅스 커맨드 $mkdir -p 방식으로 디렉토리 생성

            train_and_save(params=params, file_path=corpus_path_lst[0], token_type=token_type, tokenizer_type=tokenizer_type, composition_type=composition_type, dummy_letter=dummy_letter, save_dir=save_dir, threads=threads)

            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"elapsed time: {elapsed_time}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--parameter_settings", type=str, default="")
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--corpus_path", type=str, default="./corpus/tokenized/")

    args = vars(parser.parse_args())
    print(args)

    param_setting = pd.read_csv(args["parameter_settings"])

    corpus_path = args["corpus_path"]
    iteration = args["iteration"]
    threads = args["threads"]

    main(iteration=iteration, threads=threads, corpus_path=corpus_path, param_setting=param_setting)
