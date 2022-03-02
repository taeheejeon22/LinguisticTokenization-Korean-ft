import fasttext
import gensim
import os

# load a fasttext model
def load_ft_model(ft_model_path: str, token_type: str, tokenizer_type: str, composition_type: str, dummy_letter: str, use_gensim: bool = False):
    print(f"token type: {token_type}")
    print(f"tokenizer type: {tokenizer_type}")
    print(f"composition type: {composition_type}")
    print(f"dummy letter: {dummy_letter}")

    # 경로 설정
    files_path = ft_model_path + "/" + "_".join([token_type, tokenizer_type, composition_type, "dummy_F" if dummy_letter == False else "dummy_T"])
    print(f"path: {files_path}")

    # 모델 로드
    model_paths = os.listdir(files_path)
    model_path = files_path + "/" + sorted(model_paths)[0]

    # if len(model_path) != 1:
    if model_path.endswith(".npy"):
        raise Exception("모델 경로 확인 요망!")

    if use_gensim == True:
        # model = FastText.load(model_path)   # gensim으로 학습한 모델 로드
        model = gensim.models.fasttext.load_facebook_model(model_path)  # official로 학습한 모델 로드

    elif use_gensim == False:
        model = fasttext.load_model(model_path) # official로 학습한 모델 로드

    print("model path:", model_path)

    return model