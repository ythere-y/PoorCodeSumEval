
import torch
from tqdm import tqdm
from bleu import bleuFromMaps
torch.cuda.set_device(0) #2对应1 1对应0 0对应2
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
test_summary_model = AutoModelForCausalLM.from_pretrained('bigcode/starcoderbase', token='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', torch_dtype=torch.float16).to('cuda')

import numpy as np
import time
import datetime
import random

from tqdm import tqdm
from data_loader import SummaryStarcoderData

seed_val = 114

tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoderbase', token='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')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

timestamp=datetime.datetime.now().strftime('%Y%m%d%H%M')

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
# 2~4
epochs = 1


# ensure a certain output when running the code
###############################################################################
# Load data
###############################################################################
valid_set=SummaryStarcoderData('/data/common/CodeSearchNet/code2nl/python/test.jsonl')
valid_loader=torch.utils.data.DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=1)
print("Loaded data!")
torch.autograd.set_detect_anomaly(True)
total_t0 = time.time()
best_acc = 0
not_increase_num = 0


total_bleu = 0
for epoch_i in range(0, epochs):
    # ========================================
    #               Validation
    # ========================================
    # after each epcoh
    cnt = 0
    print("")
    print("Running Validation...")

    t0 = time.time()

    # Tracking variables 
    total_eval_accuracy = 0
    total_pred_accuracy = 0
    total_name_ok_acc = 0
    total_eval_loss = 0
    # Evaluate data for one epoch
    cnt = 0
    pred = []
    ground_truth = []
    test_summary_model.eval()
    org_tokens = []
    # result_file = open('result_llama_16.txt', 'w')
    for step, batch in enumerate(tqdm(valid_loader)):
        
        if step > 10:
            break
        cnt += 1
        # 数据
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        target_ids = batch[2].to(device)
        target_mask = batch[3].to(device)
        ground_truth_ids = list(target_ids.cpu().numpy())
        ground_truth.extend(ground_truth_ids)

        org_tokens.extend(input_ids.cpu().numpy())

        with torch.no_grad():
            new_decode_summary = test_summary_model.generate(input_ids=input_ids, attention_mask = input_mask, do_sample=False, max_new_tokens = 300, repetition_penalty=1)

        top_preds = list(new_decode_summary[:, input_ids.shape[1]:].cpu().numpy())
        pred.extend(top_preds)


    org_tokens = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in org_tokens]
    pred = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred]
    ground_truth = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in ground_truth]


    for ii in range(len(pred)):
        print('------------------')
        pred[ii] = [pred[ii].lstrip('\n').lstrip(' ').split('\n')[0].strip('"""')]
        ground_truth[ii] = [ground_truth[ii]]
        print(org_tokens[ii])
        print(pred[ii])
        print(ground_truth[ii])
        
    bleu2 = bleuFromMaps(ground_truth, pred)[0]

    print('new input bleu:', bleu2)
    
