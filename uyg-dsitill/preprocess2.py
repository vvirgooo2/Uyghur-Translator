import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="uig_Arab")
tokenizer_target = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="eng_Latn")
# prepare
from datasets import load_dataset
raw_train_dataset = load_dataset("allenai/nllb", "eng_Latn-uig_Arab", cache_dir='/home/zzy/nfsdata/nllb')
# raw_train_dataset['train'] = raw_train_dataset['train'].select([0,1,2])

def tokenize_function_train2(examples: dict):
    return tokenizer_target([d['eng_Latn'] for d in examples['translation']], padding="max_length", truncation=True)

raw_train_dataset = raw_train_dataset.remove_columns(['laser_score', 'source_sentence_lid','target_sentence_lid','source_sentence_source', 'source_sentence_url', 'target_sentence_source', 'target_sentence_url'  ])
tokenized_train_dataset_decoding = raw_train_dataset.map(tokenize_function_train2, batched=True)

tokenized_train_dataset_decoding = tokenized_train_dataset_decoding.remove_columns(['translation'])

# labels = tokenized_train_dataset_decoding['train']['input_ids']

tokenized_train_dataset_decoding.save_to_disk('/home/zzy/nfsdata/nllb-token-output')

