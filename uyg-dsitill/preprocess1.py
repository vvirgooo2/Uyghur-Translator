import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="uig_Arab")
tokenizer_target = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="eng_Latn")
# prepare
from datasets import load_dataset
raw_train_dataset = load_dataset("allenai/nllb", "eng_Latn-uig_Arab", cache_dir='/home/zzy/nfsdata/nllb')
# raw_train_dataset['train'] = raw_train_dataset['train'].select([0,2,3])

def tokenize_function_train(examples: dict):
    return tokenizer([d['uig_Arab'] for d in examples['translation']],padding="max_length", truncation=True)

def tokenize_function_train2(examples: dict):
    return tokenizer_target([d['eng_Latn'] for d in examples['translation']], padding="max_length", truncation=True)

raw_train_dataset = raw_train_dataset.remove_columns(['laser_score', 'source_sentence_lid','target_sentence_lid','source_sentence_source', 'source_sentence_url', 'target_sentence_source', 'target_sentence_url'  ])

tokenized_train_dataset_encoding = raw_train_dataset.map(tokenize_function_train, batched=True)


tokenized_train_dataset_encoding = tokenized_train_dataset_encoding.remove_columns(['translation'])


print(tokenized_train_dataset_encoding.column_names)

tokenized_train_dataset_encoding.save_to_disk('/home/zzy/nfsdata/nllb-token-input')

