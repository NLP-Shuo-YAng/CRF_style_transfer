import sys
sys.path.append('..')
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch.utils.data as Data
import torch
from seq2seq import *
from transformers import BertTokenizer
def main():
    max_length=20
    f=open('../Data/yelp/test.pos')
    lines=f.readlines()
    tensors=lines2tensors(lines, max_length=max_length)
    dataset = Data.TensorDataset(tensors, tensors)
    data_iter=torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    device=torch.device('cuda')
    generator=Transformer()
    generator=generator.to(device)
    generator.load_state_dict(torch.load('transformers.pth'))
    generator.eval()
    f.close()
    f=open('result.txt','w')
    for src, _ in data_iter:
        SOS=101
        text=[]
        input_words=torch.tensor([SOS for _ in range(src.size(0))]).view(1,-1).to(device)
        words=input_words
        for i in range(max_length-1):
            output=generator(src.t().to(device), input_words.view(i+1,-1).to(device))
            words=output.argmax(-1).view(1,-1)
            input_words=torch.cat([input_words, words], 0)
            text.append(words.clone())
        text=torch.cat(text, 0).t()
        for i in range(src.size(0)):
            f.write(tokenizer.decode(src[i].cpu().numpy().tolist(), clean_up_tokenization_spaces=False, skip_special_tokens=True)+'\t'+\
                tokenizer.decode(text[i].cpu().numpy().tolist(), clean_up_tokenization_spaces=False, skip_special_tokens=True)+'\n')
    f.close()
    return
if __name__=='__main__':
    main()
