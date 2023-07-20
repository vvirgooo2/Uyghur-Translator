import os
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data.loader_getter import prepare_data
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5Model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW
from transformers import get_scheduler
import torch.nn.functional as F
import logging
import time
import socket

print(torch.cuda.is_available())

def prepare_model(config):    
    t5config = T5Config.from_pretrained('t5-small')
    t5config.vocab_size = config['vocab_size']
    model = T5ForConditionalGeneration(t5config)
    model.init_weights()
    print('init T5 weights')
    model_teacher = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    maindevice = config['gpus'][0]
    model = nn.DataParallel(model, device_ids=config['gpus']).to(f'cuda:{maindevice}')
    model_teacher = nn.DataParallel(model_teacher, device_ids=config['gpus']).to(f'cuda:{maindevice}')
    
    return {
        'teach': model_teacher,
        'stu': model
    }


if __name__ == '__main__':
    #config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="the path to config (.yaml) file")
    parser.add_argument('--taskname', type=str, help="task name")
    parser.add_argument('--debug', type=int, default=0, help="task name")
    args = parser.parse_args()
    
    import yaml

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config['debug'] = args.debug
    
    # 获取当前时间戳
    timestamp = int(time.time())
    timestamp = time.strftime("%Y.%m%d.%H:%M:%S", time.localtime(timestamp))
    # 获取设备主机名
    hostname = socket.gethostname()

    # 组合设备和时间信息，生成唯一的字符串
    stamp = f"{hostname}_{timestamp}"
    
    # log 
    log_dir = f'logs/[{args.taskname}]-{stamp}'
    writer = SummaryWriter(log_dir=log_dir, comment=args.taskname)
    logger = logging.getLogger(args.taskname)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()  # 输出到控制台
    file_handler = logging.FileHandler(f'{log_dir}/log.log')  # 写入文件
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    
    # model train
    maindevice = config['gpus'][0]
    dataloader_dict = prepare_data(config)
    model_dict = prepare_model(config)
    
    optimizer = AdamW(model_dict['stu'].parameters(), lr=config['lr'])
    num_epochs = config['num_epochs']
    num_training_steps = len(dataloader_dict['train'])*num_epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )
    train_dataloader = dataloader_dict['train']
    val_dataloader = dataloader_dict['val']
    test_dataloader = dataloader_dict['test']
    
    
    model_dict['stu'].train()
    model_dict['teach'].eval()
    leniter = len(train_dataloader)
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="uig_Arab")
    
    
    for epoch in range(num_epochs):
        print('Epoch ', epoch, 'start')
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(f'cuda:{maindevice}') for k, v in batch.items()}
            # only need input
            outputs = model_dict['stu'](batch['input_ids'],batch['attention_mask'], batch['labels'])
            teach_out = model_dict['teach'](batch['input_ids'],batch['attention_mask'], batch['labels']).detach() 
            loss = outputs.loss + 0.3 * F.mse_loss(outputs.logits, teach_out.logits)
            loss.backward()
            logger.info('Epoch {:d}/{:d},  Iter {:d}/{:d} Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(leniter), loss.item()))
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            writer.add_scalar('Loss/train', loss, global_step=epoch * len(train_dataloader) + i)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Loss/lr', current_lr, global_step=epoch * len(train_dataloader) + i)
           
        # validation
        import evaluate
        metric = evaluate.load("chrf")
        num_training_steps = len(val_dataloader)
        for batch in val_dataloader:
            label = batch["sentence_eng_Latn"]
            batch = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
            batch = {k: v.to(f'cuda:{maindevice}') for k, v in batch.items()}
            output = model_dict['stu'].generate(**batch, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
            predictions = tokenizer.batch_decode(output, skip_special_tokens=True)
            # print(tokenizer.batch_decode(batch['input_ids']),predictions,label)
            metric.add_batch(predictions=predictions, references=label)
        score = metric.compute(word_order=2)
        logger.info(score)
        writer.add_scalar('Score/val', score['score'], global_step=epoch)
        
        # save model per 5 epoch
        if (epoch+1)%config['save_frequency'] == 0:
            torch.save({
                'epoch': epoch,
                "model_state_dict": model_dict['stu'].state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
            },
                "checkpoint/saved_model{:d}.pt".format(epoch))
        logger.info("save to checkpoint/saved_model{:d}.pt".format(epoch))
    
    # end test
    metric = evaluate.load("chrf")
    num_training_steps = len(test_dataloader)
    for batch in test_dataloader:
        label = batch["sentence_eng_Latn"]
        batch = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        batch = {k: v.to(f'cuda:{maindevice}') for k, v in batch.items()}
        output = model_dict['stu'].generate(**batch, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
        predictions = tokenizer.batch_decode(output, skip_special_tokens=True)
        # print(tokenizer.batch_decode(batch['input_ids']),predictions,label)
        metric.add_batch(predictions=predictions, references=label)
    score = metric.compute(word_order=2)
    logger.info(score)
    
    