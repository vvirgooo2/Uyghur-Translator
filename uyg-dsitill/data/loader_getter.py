from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def prepare_data(config):
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="uig_Arab")
    tokenizer_target = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="eng_Latn")


    raw_eval_dataset = load_dataset("Muennighoff/flores200", "eng_Latn-uig_Arab",  cache_dir=config['flores200_dir'])

    def tokenize_function(examples):
        return tokenizer(examples["sentence_uig_Arab"], padding="max_length", truncation=True)

    def tokenize_function_target(examples: dict):
        return tokenizer_target( examples["sentence_eng_Latn"], padding="max_length", truncation=True)


    tokenized_eval_dataset = raw_eval_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset_target = raw_eval_dataset.map(tokenize_function_target, batched=True)
    labels = tokenized_eval_dataset_target['dev']['input_ids']
    tokenized_eval_dataset['dev'] = tokenized_eval_dataset['dev'].add_column(name='labels', column=labels)

    labels = tokenized_eval_dataset_target['devtest']['input_ids']
    tokenized_eval_dataset['devtest'] = tokenized_eval_dataset['devtest'].add_column(name='labels', column=labels)
    
    tokenized_eval_dataset['dev'] = tokenized_eval_dataset['devtest'].remove_columns(['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink','sentence_uig_Arab'])
    tokenized_eval_dataset['devtest'] = tokenized_eval_dataset['devtest'].remove_columns(['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink','sentence_uig_Arab'])
    
    tokenized_eval_dataset.set_format("torch")
    print(tokenized_eval_dataset['dev'].column_names)
    print(tokenized_eval_dataset['devtest'].column_names)

    from torch.utils.data import DataLoader
    val_dataloader = DataLoader(
        tokenized_eval_dataset['dev'], config['batch_size'], shuffle=False
    )
    test_dataloader = DataLoader(
        tokenized_eval_dataset['devtest'], config['batch_size'], shuffle=False
    )

    dataset_token_train_uig = load_from_disk(config['token_uig_dir'])
    print('Uig data Loaded')
    dataset_token_train_eng = load_from_disk(config['token_eng_dir'])
    print('Eng data Loaded')
    
    dataset_token_train_eng['train'] = dataset_token_train_eng['train'].remove_columns(['attention_mask'])
    dataset_token_train_eng['train'] = dataset_token_train_eng['train'].rename_column('input_ids', 'labels')

    dataset_new = concatenate_datasets([dataset_token_train_uig['train'], dataset_token_train_eng['train']], axis=1)
    print('Uig-Eng data Generated')
    dataset_new.set_format("torch")

    if config['debug']==1:
        dataset_new = dataset_new.select(range(100))
        
    train_dataloader = DataLoader(
        dataset_new, config['batch_size'], shuffle=True
    )
    
    return {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }
