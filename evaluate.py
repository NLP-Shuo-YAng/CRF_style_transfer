from transformers import BertTokenizer
from model.classifier import TextCNN
import torch
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def evaluate(classifier_save_pth, pth, ref_pth=None):
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    f=open(pth)
    lines=f.readlines()
    src_lines=[]
    out_lines=[]
    for line in lines:
        line=line.split('\t')
        src_lines.append(tokenizer.encode(line[0]))
        out_lines.append(tokenizer.encode(line[1]))
    f.close()
    if ref_pth:
        ref_lines=[]
        f=open(ref_pth)
        lines=f.readlines()
        for line in lines:
            ref_lines.append(tokenizer.encode(line))
        f.close()
    device=torch.device('cuda')
    classifier=TextCNN(tokenizer.vocab_size)
    classifier.load_state_dict(torch.load(classifier_save_pth))
    classifier=classifier.to(device)
    classifier.eval()
    acc=0
    for line in out_lines:
        if len(line)<5:
            for _ in range(5-len(line)):
                line.append(0)
        acc_result=int(classifier(torch.tensor(line).view(1,-1).to(device)).argmax(-1))
        acc+=acc_result
        if acc_result==0:
            print(tokenizer.decode(line, skip_special_tokens=True))
    print('Accuracy: ', round(acc/len(out_lines)*100, 3), '%')
    self_bleu=[]
    ref_bleu=[]
    smooth=SmoothingFunction()
    for i in range(len(out_lines)):
        src_line=tokenizer.tokenize(tokenizer.decode(src_lines[i], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        out_line=tokenizer.tokenize(tokenizer.decode(out_lines[i], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        self_bleu.append(float(sentence_bleu([src_line], out_line, smoothing_function=smooth.method1)))
        if ref_pth:
            ref_line=tokenizer.tokenize(tokenizer.decode(ref_lines[i], skip_special_tokens=True, clean_up_tokenization_spaces=False))
            ref_bleu.append(float(sentence_bleu([ref_line], out_line, smoothing_function=smooth.method1)))
    if ref_pth:
        ref_bleu_miu, ref_bleu_co=confi(ref_bleu)
        print('ref-BLEU:', round(ref_bleu_miu*100,2), '+-',round(ref_bleu_co*100,2) )
    self_bleu_miu, self_bleu_co=confi(self_bleu)
    print('self-BLEU:', round(self_bleu_miu*100,2), '+-',round(self_bleu_co*100,2) )
    return

def mean(v_list):
    a=0
    for v in v_list:
        a+=v
    return a/len(v_list)
def confi(v_list):
    n=len(v_list)
    miu=mean(v_list)
    Sn=0
    for i in v_list:
        Sn+=math.pow((i-miu),2)
    Sn/=(n*(n-1))
    co=1.96*math.sqrt(Sn)
    return miu, co
def main():
    classifier_save_pth='classifier.pth'
    pth='parallel/yelp_neg2pos.txt'
    ref_pth='./Data/Amazon/reference.neg'
    evaluate(classifier_save_pth, pth, ref_pth=ref_pth)
    return

if __name__=='__main__':
    main()
