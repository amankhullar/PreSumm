import os
import sys

import numpy as np

from rouge import Rouge

def main(cand_path, ref_path):
    try:
        with open(cand_path, 'r') as f:
            cand_sums = f.readlines()
    except Exception as e:
        print("Cannot open candidate summaries with error " + str(e))
    try:
        with open(ref_path, 'r') as f:
            ref_sums = f.readlines()
    except Exception as e:
        print("Cannot open reference summaries with error " + str(e))
    
    rouge = Rouge()
    rouge_scores = rouge.get_scores(cand_sums, ref_sums, avg=True)

    print("Rouge 1 score is : {0:.3f}".format(rouge_scores['rouge-1']['f']*100))
    print("Rouge 2 score is : {0:.3f}".format(rouge_scores['rouge-2']['f']*100))
    print(rouge_scores)

if __name__ == "__main__":
    cand_path = "/home/aman_khullar/PreSumm/logs/test_abs_bert_cnnd_res.154000.candidate"
    ref_path = "/home/aman_khullar/PreSumm/logs/test_abs_bert_cnnd_res.154000.gold"
    main(cand_path, ref_path)

