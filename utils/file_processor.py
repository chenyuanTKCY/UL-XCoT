########################################################################
#
# @author : Chenyuan Zhang
# @when : Winter Semester 2025/2026
# @where : Harbin Institute of Technology
# @title : File Processor
# @component: utils
# @file : file_processer.py
#
########################################################################

from typing import List, Tuple, Optional
from utils.config import polymath_LANG_LIST, language_sample_dir, mmlu_LANG_LIST
import os
import sys
import argparse
import pickle


def read_file(file_path: str)->str:
    with open(file_path, 'r') as f:
        return f.read()
    return ""


def read_file_line_by_line(lang_id:int, file_path: str, encoding: str = 'utf-8')->dict:
    input_data = []
    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            data = line.strip().split("\t")
            try:
                if len(data[0]) < 12:
                    input_data.append({"lang_id": lang_id, "text": data[1], "answer": data[2]})
                else:
                    input_data.append({"lang_id": lang_id, "text": data[0], "answer": data[1]})
            except Exception as e:
                print(e)
                continue
        return input_data
    return ""

def read_allfiles_in_dir_byline(dir_path: str, file_format: str)->List[str]:
    all_result = read_allfiles_in_dir(dir_path, file_format)
    # print(all_result)
    result = []
    # problem_n , lang_n, 1
    for line in range(len(all_result[1])):
        temp_result = []
        for lang in range(len(all_result)):
            temp_result.append([all_result[lang][line]])
        result.append(temp_result)
    
    return result

def read_pkl_file(file_path: str)->Optional[object]:
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    return None

def read_allfiles_in_dir(dir_path: str, file_format: str)->List[str]:
    all_result = []
    for idx, filename in enumerate(sorted(os.listdir(dir_path))):
        if filename.endswith(file_format):
            file_path = os.path.join(dir_path, filename)
            content = read_file_line_by_line(idx, file_path)
            all_result.append(content)
    return all_result


def read_polymath(mode:str = "sample", diff:str = "low", translate_to_en:bool = False):
    if mode == "sample":
        language_all_sample = []
        for lang_id, lang_name in enumerate(polymath_LANG_LIST):
            language_sample = []
            for diff_d, diff_level in enumerate(["low", "medium", "high", "top"]):
                file_path = language_sample_dir + f"/{lang_name}/{diff_level}.tsv"
                language_matrix_set = read_file_line_by_line(lang_id, file_path)
                language_sample.extend(language_matrix_set)
                # print("!")
                # if len(language_sample) == 124:
                #     print(f"Warning: language {lang_name} has less than 125 samples in {diff_level} difficulty level.")
                # else:
                #     print(len(language_sample))
            language_all_sample.append(language_sample)

        return language_all_sample
    elif mode == "translate":
        language_all_sample = {}
        for diff_d, diff_level in enumerate(["low", "medium", "high", "top"]):
            language_sample = []
            for lang_id, lang_name in enumerate(polymath_LANG_LIST):    
                file_path = language_sample_dir + f"/{lang_name}/{diff_level}.tsv"
                language_matrix_set = read_file_line_by_line(lang_id, file_path)
                language_sample.append(language_matrix_set)
            language_all_sample[diff_level] = language_sample
        return language_all_sample
    else:
        language_all_query_set = []
        # if translate_to_en:
        #     language_sample_dir = "./dataset/polymath/translate/"
        for lang_id, lang_name in enumerate(polymath_LANG_LIST):
            file_path = language_sample_dir + f"/{lang_name}/{diff}.tsv"
            language_query_set = read_file_line_by_line(lang_id, file_path)
            if mode == "sample_single":
                new_language_query_set = [row for row in language_query_set]
            else:
                new_language_query_set = [[row] for row in language_query_set]
            language_all_query_set.append(new_language_query_set)
        transposed = [list(row) for row in zip(*language_all_query_set)]
        return transposed

def read_MMLU(mode:str = "sample"):
    if mode == "sample":
        language_all_sample = []
        for lang_id, lang_name in enumerate(mmlu_LANG_LIST):
            language_sample = []
            # for diff_d, diff_level in enumerate(["low", "medium", "high", "top"]):
            file_path = "dataset/MMLU-ProX-Lite_2col_tsv_by_lang" + f"/{lang_name}/test.tsv"
            language_matrix_set = read_file_line_by_line(lang_id, file_path)
            language_sample.extend(language_matrix_set)
            language_all_sample.append(language_sample)

        return language_all_sample
    else:
        language_all_query_set = []
        for lang_id, lang_name in enumerate(mmlu_LANG_LIST):
            file_path = "dataset/MMLU-ProX-Lite_2col_tsv_by_lang" + f"/{lang_name}/test.tsv"
            language_query_set = read_file_line_by_line(lang_id, file_path)
            if mode == "sample_single":
                new_language_query_set = [row for row in language_query_set]
            else:
                new_language_query_set = [[row] for row in language_query_set]
            language_all_query_set.append(new_language_query_set)
        transposed = [list(row) for row in zip(*language_all_query_set)]
        return transposed

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(prog = "file_processor",description="File Processor")
    argparser.add_argument("--file_path", type=str, help="The path of the file to be read.")
    argparser.add_argument("--dir_path", type=str, help="The directory path to be read.")
    argparser.add_argument("--file_format", type=str, help="The file format to be read.")
    args = argparser.parse_args()

    # result = read_allfiles_in_dir_byline(args.dir_path, args.file_format)
    
    
    
