########################################################################
#
# @author : Chenyuan Zhang
# @when : Winter Semester 2025/2026
# @where : Harbin Institute of Technology
# @title : Config settings
# @component: utils
# @file : config.py
#
########################################################################

import os
import torch
from typing import Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

ms_save_dir = os.environ.get("ULXCOT_MS_SAVE_DIR", "./results/linguistic_difference/")
mgsm_output_path = os.environ.get("ULXCOT_MGSM_OUTPUT_PATH", "./outputs/mgsm/")
# polymath_output_path = "./dataset/polymath/output_orign/"
# polymath_output_path = "./dataset/polymath/qwen2.5_3b/output/"
# polymath_output_path = "./dataset/polymath/ds7b/output/"
# polymath_output_path = "./dataset/polymath/ds1.5b/output/"
# polymath_output_path = "./dataset/polymath/ds1.5b/output_consistency/"
# polymath_output_path = "./dataset/polymath/ds1.5b/new_output/"
polymath_output_path = os.environ.get("ULXCOT_POLYMATH_OUTPUT_PATH", "./outputs/polymath/")

gamma_save_dir = os.environ.get("ULXCOT_GAMMA_SAVE_DIR", "./results/linguistic_coefficency/")
ma_diff_save_dir = os.environ.get("ULXCOT_MA_DIFF_SAVE_DIR", "./results/logic_inference/")
language_model_dir = os.environ.get(
    "ULXCOT_MODEL_PATH",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
)
# language_sample_dir = "./dataset/mgsm/input"
language_sample_dir = "./dataset/polymath/input"
# language_query_dir = "./dataset/mgsm/input"

mgsm_LANG_LIST = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
polymath_LANG_LIST = ['ar', 'bn', 'de', 'en', 'es', 'fr', 'id', 'it', 'ja', 'ko', 'ms', 'pt', 'ru', 'sw', 'te', 
                      'th', 'vi', 
                      'zh']
mmlu_LANG_LIST = langs = [
    "af","ar","bn","cs","de","en","es","fr","hi","hu","id","it","ja","ko",
    "mr","ne","pt","ru","sr","sw","te","th","uk","ur","vi","wo","yo","zh","zu"
]
mmlu_LANG_DICT = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bn": "Bengali",
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mr": "Marathi",
    "ne": "Nepali",
    "pt": "Portuguese",
    "ru": "Russian",
    "sr": "Serbian",
    "sw": "Swahili",
    "te": "Telugu",
    "th": "Thai",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "wo": "Wolof",
    "yo": "Yoruba",
    "zh": "Chinese",
    "zu": "Zulu",
}
mgsm_LANG_DICT = {
    "bn": "Bengali", 
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "ru": "Russian",
    "sw": "Swahili",
    "te": "Telugu",
    "th": "Thai",
    "zh": "Chinese"
}

polymath_LANG_DICT = {
    'ar': 'Arabic',
    'bn': 'Bengali', 
    'de': 'German',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'id': 'Indonesian',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ms': 'Malay',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'sw': 'Swahili',
    'te': 'Telugu',
    "th": "Thai",
    "vi": "Vietnamese",
    'zh': 'Chinese'
}

polymath_extra_LANG_DICT = {
    "th": "Thai",
    "vi": "Vietnamese",
}
polymath_reverse = {
  'Arabic':      {'code': 'ar', 'idx1': 1,  'idx0': 0},
  'Bengali':     {'code': 'bn', 'idx1': 2,  'idx0': 1},
  'German':      {'code': 'de', 'idx1': 3,  'idx0': 2},
  'English':     {'code': 'en', 'idx1': 4,  'idx0': 3},
  'Spanish':     {'code': 'es', 'idx1': 5,  'idx0': 4},
  'French':      {'code': 'fr', 'idx1': 6,  'idx0': 5},
  'Indonesian':  {'code': 'id', 'idx1': 7,  'idx0': 6},
  'Italian':     {'code': 'it', 'idx1': 8,  'idx0': 7},
  'Japanese':    {'code': 'ja', 'idx1': 9,  'idx0': 8},
  'Korean':      {'code': 'ko', 'idx1': 10, 'idx0': 9},
  'Malay':       {'code': 'ms', 'idx1': 11, 'idx0': 10},
  'Portuguese':  {'code': 'pt', 'idx1': 12, 'idx0': 11},
  'Russian':     {'code': 'ru', 'idx1': 13, 'idx0': 12},
  'Swahili':     {'code': 'sw', 'idx1': 14, 'idx0': 13},
  'Telugu':      {'code': 'te', 'idx1': 15, 'idx0': 14},
  'Chinese':     {'code': 'zh', 'idx1': 16, 'idx0': 15},
}
polymath_LANG_NOTE_DICT = {
    'ar': 'ملاحظة: يُرجى كتابة الإجابة النهائية بصيغة $\\boxed{{}}$.',
    'bn': 'বিঃদ্রঃ অনুগ্রহ করে চূড়ান্ত উত্তরটি $\\boxed{{}}$ ফরম্যাটে লিখুন।',
    'de': 'Hinweis: Bitte geben Sie die endgültige Antwort im Format $\\boxed{{}}$ an.',
    'en': 'Note: Please put the final answer in the $\\boxed{{}}$ format.',
    'es': 'Nota: Por favor, escribe la respuesta final en el formato $\\boxed{{}}$.',
    'fr': 'Remarque : Veuillez donner la réponse finale au format $\\boxed{{}}$.',
    'id': 'Catatan: Harap tuliskan jawaban akhir dalam format $\\boxed{{}}$.',
    'it': 'Nota: Inserisci la risposta finale nel formato $\\boxed{{}}$.',
    'ja': '注意：最終的な答えは $\\boxed{{}}$ の形式で書いてください。',
    'ko': '주의: 최종 답은 $\\boxed{{}}$ 형식으로 써 주세요.',
    'ms': 'Nota: Sila tulis jawapan akhir dalam format $\\boxed{{}}$.',
    'pt': 'Observação: Por favor, escreva a resposta final no formato $\\boxed{{}}$.',
    'ru': 'Примечание: пожалуйста, запишите окончательный ответ в формате $\\boxed{{}}$.',
    'sw': 'Kumbuka: Tafadhali andika jibu la mwisho katika umbizo la $\\boxed{{}}$.',
    'te': 'గమనిక: దయచేసి తుది సమాధానాన్ని $\\boxed{{}}$ రూపంలో వ్రాయండి.',
    'zh': '注意：请将最终答案写成 $\\boxed{{}}$ 的形式。'
}


def model_initialization()-> Tuple :
    '''
        This function is used to initialize the language model and tokenizer
    '''

    _model = AutoModelForCausalLM.from_pretrained(
    language_model_dir,
    output_hidden_states=True,
    dtype=torch.bfloat16,
    device_map="cuda:0"
    )
    _tokenizer = AutoTokenizer.from_pretrained(language_model_dir)
    return (_model, _tokenizer)


def model_initialization_parallel(model_path_use, sampling_size:int = 3, mode:str = 'same', indicated_device:int = 0)-> Tuple:
    '''
        This function is used to initialize the language model and tokenizer in parallel
    '''
    models = []
    tokenizers = []
    if mode == 'same':
        device_ids = available_gpus_same(sampling_size, indicated_device= indicated_device)
    else:
        device_ids = available_gpus(sampling_size)
    print( f"Available device ids for model initialization: {device_ids}" )
    model_path = model_path_use
    for device_id in device_ids:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device_id,
            trust_remote_code=True
        )
        models.append(model)
        tokenizers.append(tokenizer)
    return (models, tokenizers)


def available_gpus(min_required_gpus: int = 3, memory_threshold: int = 100) -> List:
    '''
        This function is used to get the available gpus
    args:
        min_required_gpus: The min number of GPUs required
    return:
        A list of available GPU device IDs
    '''
    available_devices = []
    count = 0
    for i in range(torch.cuda.device_count()):

        if len(available_devices) >= min_required_gpus:
            break

        try:

            memory_used_mb = torch.cuda.memory_allocated(i) / (1024 * 1024)
            memory_reserved_mb = torch.cuda.memory_reserved(i) / (1024 * 1024)
            
            if memory_used_mb < memory_threshold and memory_reserved_mb < memory_threshold:
                available_devices.append(f"cuda:{i}")
                
        except Exception as e:
            print(f"Warning: Could not check GPU {i}: {e}")
            continue
    return available_devices


def available_gpus_same(min_required_gpus: int = 3, indicated_device:int = 0) -> List:
    '''
        This function is used to get the available gpus
    args:
        min_required_gpus: The min number of GPUs required
    return:
        A list of available GPU device IDs
    '''
    available_devices = []
    count = 0
    for i in range(min_required_gpus):
        available_devices.append(f"cuda:{indicated_device}")

    return available_devices
