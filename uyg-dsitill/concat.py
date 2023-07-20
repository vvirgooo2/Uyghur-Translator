from datasets import load_from_disk, load_dataset, concatenate_datasets

dataset_token_train_uig = load_from_disk('/home/zzy/nfsdata/nllb-token-uig')
print(dataset_token_train_uig['train'].column_names)
print(dataset_token_train_uig['train'].num_rows)

dataset_token_train_eng = load_from_disk('/home/zzy/nfsdata/nllb-token-eng')
print(dataset_token_train_eng['train'].column_names)
print(dataset_token_train_eng['train'].num_rows)
dataset_token_train_eng['train'] = dataset_token_train_eng['train'].remove_columns(['attention_mask'])
dataset_token_train_eng['train'] = dataset_token_train_eng['train'].rename_column('input_ids', 'labels')

print('1')
dataset_new = concatenate_datasets([dataset_token_train_uig['train'], dataset_token_train_eng['train']], axis=1)
print(dataset_new)