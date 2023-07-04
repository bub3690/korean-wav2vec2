import torch
import queue
import os
import random
import warnings
import time
import json
import argparse
from glob import glob
import numpy as np
from datasets import load_metric

# from modules.preprocess import preprocessing, remove_special_characters
# from modules.trainer import trainer

# from modules.model import build_model
# from modules.data import load_dataset, DataCollatorCTCWithPadding
# from modules.inference import single_infer

from torch.utils.data import DataLoader

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import TrainingArguments, Trainer

DATASET_PATH="../data/t2-conf"

def bind_model(model,processor, optimizer=None):
    def save(path, *args, **kwargs):
        print("save!!! " + path)
        model.save_pretrained(path)
        processor.save_pretrained(path)
        print('Model saved')

    def load(path, *args, **kwargs):
        print("load!!! " + path)
        processor.from_pretrained(path)
        model.from_pretrained(path)
        model.to("cuda")
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model, processor)
    
    

def prepare_dataset(batch):
    signal = [ np.memmap(sample, dtype='h', mode='r').astype('float32') for sample in batch["path"] ]
    #batch["sampling_rate"] = [16_000] * len(signal)
    batch["input_values"] = processor(signal, sampling_rate=16_000).input_values
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

cer_metric = load_metric('cer')
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    print("pred: ",pred_str)
    print("ref : ",label_str)

    cer = cer_metric.compute(references=label_str, predictions=pred_str)

    return {"cer": cer}

def inference(path, model, processor, **kwargs):
    model.eval()

    results = []
    for i in glob(os.path.join(path, '*')):
        print("===========================")
        print('filename : ', i.split('/')[-1])
        print(i)
        result = single_infer(model, i, processor)[0]
        print("result : " , result)
        results.append(
            {
                'filename': i.split('/')[-1],
                'text': result
            }
        )
    return sorted(results, key=lambda x: x['filename'])



if __name__ == '__main__':
    print(torch.__version__)
    args = argparse.ArgumentParser()

    #config = args.parse_args()
    warnings.filterwarnings('ignore')

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #torch.cuda.set_device(0)
    print("device : ",device)
    if hasattr(config, "num_threads") and int(config.num_threads) > 0:
        torch.set_num_threads(config.num_threads)
    
    # tokenizer & feature_extractor & processor
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = build_model(config, processor, device)
    print(f'Load Model is done')
    
    #bind_model(model,processor)
        
    if config.mode == 'train':
        config.dataset_path = os.path.join(DATASET_PATH, 'train', 'train_data')
        label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
        preprocessing(label_path, os.getcwd())
        dataset = load_dataset(os.path.join(os.getcwd(), 'transcripts.txt'))

        dataset = dataset.map(remove_special_characters)

        dataset = dataset.flatten_indices()
        #dataset = dataset.map(speech_file_to_array_fn, remove_columns=dataset.column_names,num_proc=10)
        #print(f'speech_file_to_array_fn is done')

        dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names,batched=True, num_proc=10)
        print(f'prepare_dataset is done')
        
        dataset = dataset.train_test_split(test_size=0.2,seed=42)
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        print(f'split Datset is done')

        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        model.freeze_feature_extractor()
        print(f'freeze_feature_extractor is done')

        training_args = TrainingArguments(
            output_dir="container_1/ckpts/",
            logging_dir = "container_1/runs/",
            group_by_length=True,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            eval_accumulation_steps=2,
            evaluation_strategy="steps",
            num_train_epochs=config.num_epochs,
            fp16=True,
            save_steps=20,
            eval_steps=200,
            logging_steps=200,
            learning_rate=4e-4,
            warmup_steps=int(0.1*1320), #10%
            save_total_limit=2,
        )
        print(f'train being!')
        
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=processor.feature_extractor,
        )

        bind_model(model,processor)
        
        print("train start")
        trainer.train()

        # nsml.save(config.num_epochs)
        # 1. wav2vec2-large-xlsr-kn  #0.3191 cer

        print('[INFO] train process is done')




