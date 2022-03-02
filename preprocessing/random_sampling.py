# 학습 데이터 용량을 줄이기 위해 코퍼스에서 문장 랜덤 샘플링

import argparse
import random
import time


def main(corpus, sampling_ratio):
    sampled_corpus = random.sample(corpus, round(len(corpus) * sampling_ratio))

    return sampled_corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--sampling_ratio", type=float, default=0.25)
    parser.add_argument("--output_path", type=str, default="")

    args = vars(parser.parse_args())
    print(args)


    start_time = time.time()

    # load a corpus
    with open(args["corpus_path"]) as f:
        corpus = f.readlines()


    # set seed
    random.seed(args["seed"])


    # random sampling
    sampled_corpus = main(corpus=corpus, sampling_ratio=args["sampling_ratio"])


    # save
    if args["output_path"] == "":   # If there is no output_path
        output_path = args["corpus_path"].split(".txt")[0] + "_sampled.txt"
    else:
        output_path = args["output_path"]

    with open(output_path, "w") as f:
        for line in sampled_corpus:
            f.write(line)


    print(f"\nsaved in: {output_path}\n")
    print("Done!\n")

    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"(elapsed time: {elapsed_time})")