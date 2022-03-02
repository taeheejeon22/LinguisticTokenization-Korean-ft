import argparse
import os
import pandas as pd


# csv 파일 로드
def get_result(csv_path: str, column_names: list):
    csv = pd.read_csv(csv_path)

    return csv[column_names]


# csv 파일 템플릿 로드
def get_template(csv_path: str):
    template = pd.read_csv(csv_path)
    return template


def main(results_path: str, column_names: list):

    csv_paths = sorted([path for path in os.listdir(results_path) if path.endswith(".csv")])

    results = list()

    for ix in range(len(csv_paths)):
        csv_path = os.path.join(results_path, csv_paths[ix])

        results.append( get_result(csv_path=csv_path, column_names=column_names) )


    mean_result = sum(results) / len(csv_paths)

    mean_result = round(mean_result, 4)

    csv_template = get_template(csv_path=csv_path)
    csv_template[column_names] = mean_result

    file_name = "_".join([results_path.split("/")[-1]] + ["average_result.csv"])
    file_name = file_name.replace("__", "_")
    save_path = os.path.join("./test_results/average", file_name)

    csv_template.to_csv( save_path, index=False)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="")

    args = vars(parser.parse_args())

    results_path = args["results_path"]

    if "sim_anal" in results_path:
        column_names = ["Similarity_Pearson's r", "Similarity_Spearman's \u03C1",
                        "Analogy Semantic Capt_3CosAdd Distance", "Analogy Semantic Gend_3CosAdd Distance", "Analogy Semantic Name_3CosAdd Distance", "Analogy Semantic Lang_3CosAdd Distance", "Analogy Semantic Misc_3CosAdd Distance", "Analogy Semantic Total_3CosAdd Distance",
                        "Analogy Syntactic Case_3CosAdd Distance", "Analogy Syntactic Tense_3CosAdd Distance", "Analogy Syntactic Voice_3CosAdd Distance", "Analogy Syntactic Form_3CosAdd Distance", "Analogy Syntactic Honr_3CosAdd Distance", "Analogy Syntactic Total_3CosAdd Distance",
                        ]
    else:
        column_names = ["Accuracy"]


    main(results_path=results_path, column_names=column_names)
