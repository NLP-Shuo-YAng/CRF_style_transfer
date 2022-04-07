import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.utils.data as Data
from transformers import BertTokenizer, BertForSequenceClassification
from LM import data_processor
from model.classifier import pretrain_classifier
from eval_classifier import TextMLP
from model.seq2seq import Transformer, Bert
from rewards import Rewards, policy_gradient_loss
from operation import edit_operations
from vocab import Tokenizer
import math
import time

def tensors(pth_src, pth_tgt, max_len=20):
    src_tensors=data_processor(pth_src, max_len)
    tgt_tensors=data_processor(pth_tgt, max_len)
    return src_tensors, tgt_tensors

def label_create(pos_tensors, classifier, lm_pth, kenlm_pth, max_len=20, max_operations_num=20, search_type='greedy_search'):
    device=torch.device('cuda')
    classifier.eval()
    classifier=classifier.to(device)
    rewards_ccl=Rewards(classifier, kenlm_pth)
    Operation=edit_operations(lm_pth)
    tgt=[]
    pro_count=0
    special_tokens=[101, 102, 0, 999, 1010, 1012, 1029]
    t_start=time.time()
    for tensor in pos_tensors:
        pro_count+=1
        print('Progress:', round(pro_count/pos_tensors.size(0)*100, 3), '%')
        if search_type=='greedy_search':
            for _ in range(max_operations_num):
                scores=[]
                actions=[]
                probabilities=[]
                for a in range(1,4):
                    for t in range(tensor.size(-1)):
                        action=[]
                        for _ in range(0, t):
                            action.append(4)
                        action.append(a)
                        for _ in range(t, tensor.size(-1)-1):
                            action.append(4)
                        action=torch.tensor(action)
                        actions.append(action)
                        if int(t) in special_tokens:
                            scores.append(float('-inf'))
                        else:
                            generated_sentence=Operation.seq_operate(tensor.clone(), action).long()
                            scores.append(float(rewards_ccl.total_rewards(generated_sentence.view(-1).to(device), tensor.clone().to(device))))
                total_score=0
                for score in scores:
                    total_score+=math.pow(math.e, score)
                for score in scores:
                    probabilities.append(math.pow(math.e, score)/total_score)
            #position=int(torch.multinomial(torch.tensor(probabilities), 1))
                position=torch.tensor(probabilities).argmax(-1)
                action=actions[position]
                generated_sentence=Operation.seq_operate(tensor.clone(), action).long()
                if early_stop(classifier, generated_sentence):
                    break
                else:
                    tensor=generated_sentence.view(-1).clone()
            tgt.append(generated_sentence.view(1,-1).clone())
        else:
            generated_sentences=None
            early_flag=False
            for _ in range(max_operations_num):
                if generated_sentences==None:
                    solution_position, generated_sentences, probabilities=viterbi(tensor.clone(), classifier, Operation, rewards_ccl, device)
                    if early_stop(classifier, generated_sentences[solution_position]):
                        early_flag=True
                        tgt.append(generated_sentences[solution_position].view(1,-1).clone())
                        break
                else:
                    new_generated_sentences=[]
                    new_probabilities=[]
                    for s in range(len(generated_sentences)):
                        generated_sentence=generated_sentences[s]
                        _, tmp_generated_sentences, tmp_probabilities=viterbi(generated_sentence.clone(), classifier, Operation, rewards_ccl, device)
                        for p in range(len(tmp_probabilities)):
                            tmp_probabilities[p]*=probabilities[s]
                        new_probabilities.append(tmp_probabilities)
                        new_generated_sentences.append(tmp_generated_sentences)
                    candi=torch.tensor(new_probabilities).t().argmax(-1)
                    generated_sentences=[]
                    probabilities=[]
                    for c in range(candi.size(-1)):
                        generated_sentences.append(new_generated_sentences[candi[c]][c])
                        probabilities.append(new_probabilities[candi[c]][c])
                    solution_position=torch.tensor(probabilities).argmax(-1)
                    if early_stop(classifier, generated_sentences[solution_position]):
                        early_flag=True
                        tgt.append(generated_sentences[solution_position].view(1,-1).clone())
                        break
            if early_flag==False:
                tgt.append(generated_sentences[solution_position].view(1,-1).clone())
    print('Finished.')
    print('Time: ', time.time()-t_start)
    return torch.cat(tgt, 0)

def viterbi(tensor, classifier, Operation, rewards_ccl, device):
    actions=[]
    scores=[]
    generated_sentences=[]
    special_tokens=[101, 102, 0, 999, 1010, 1012, 1029]
    for a in range(1,4):
        for t in range(tensor.size(-1)):
            action=[]
            for _ in range(0, t):
                action.append(4)
            action.append(a)
            for _ in range(t, tensor.size(-1)-1):
                action.append(4)
            action=torch.tensor(action)
            actions.append(action)
            if int(t) in special_tokens:
                scores.append(float('-inf'))
                generated_sentence=Operation.seq_operate(tensor.clone(), action).long()
                generated_sentences.append(generated_sentence.view(1,-1).clone())
            else:
                generated_sentence=Operation.seq_operate(tensor.clone(), action).long()
                generated_sentences.append(generated_sentence.view(1,-1).clone())
                '''
                print('------------------------')
                print(generated_sentence)
                print(tensor)
                print('------------------------')
                '''
                scores.append(float(rewards_ccl.total_rewards(generated_sentence.view(-1).to(device), tensor.clone().to(device))))
    probabilities=[]
    total_score=0
    for score in scores:
        total_score+=math.pow(math.e, score)
    for score in scores:
        probabilities.append(math.pow(math.e, score)/total_score)
    position=torch.tensor(probabilities).argmax(-1)
    return position, generated_sentences, probabilities

def common_seq(seq_A, seq_B, max_len):
    device=torch.device('cuda')
    final_action=[]
    for i in range(seq_A.size(-1)):
        if int(seq_A[i])==1:
            seq_B=torch.cat([seq_B[:i], seq_B[i+1:], torch.tensor([0]).to(device)], -1)
        elif int(seq_A[i])==2:
            seq_B=torch.cat([seq_B[:i], torch.tensor([4]).to(device), seq_B[i:-1]], -1)
    for i in range(max_len):
        if seq_A[i]==seq_B[i]:
            final_action.append(int(seq_A[i]))
        else:
            if int(seq_A[i])==4:
                if int(seq_B[i])==0:
                    final_action.append(4)
                else:
                    final_action.append(int(seq_B[i]))
            elif int(seq_A[i])==0:
                final_action.append(int(seq_B[i]))
            else:
                final_action.append(int(seq_A[i]))
    return final_action
def early_stop(classifier, generated_sentence):
    output=F.softmax(classifier(generated_sentence), -1).view(-1)
    #print(float(output[1]))
    if float(output[1])>0.95:
        return True
    else: return False
def maskNLLLoss(source, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(source, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    return loss, nTotal.item()

def mask_batch(batch):
    mask = binaryMatrix(batch)
    mask = torch.BoolTensor(mask)
    return mask

def binaryMatrix(l, PAD_token=0):
    m=[]
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def remove_pad(batch, pad=0):
    batch=batch.t()
    for i in range(batch.size(0)):
        flag=True
        line=batch[i]
        for w in line:
            if int(w)!=0:
                flag=False
        if flag:
            break
    return batch[:i,:].t()

def label_output(tensors, tgt, LM_pth):
    device=torch.device('cuda')
    output_tensors=[]
    Operation=edit_operations(LM_pth)
    for i in range(tensors.size(0)):
        print('------')
        print(tensors[i])
        print(tgt[i])
        word_seq=Operation.seq_operate(tensors[i].to(device), tgt[i].to(device))
        output_tensors.append(word_seq)
    return output_tensors


def train(tokenizer, pos_tensors, generator, classifier, epoch_num, save_pth, lm_pth, kenlm_pth, max_length=20, \
          clip=50, batch_size=64, show_step=5, eval_step=100):
    device=torch.device('cuda')
    tgt_tensors=label_create(pos_tensors, classifier, lm_pth, kenlm_pth, max_len=max_length)
    f=open('output.txt','w')
    for i in range(tgt_tensors.size(0)):
        f.write(tokenizer.decode(pos_tensors[i].cpu().numpy().tolist(), clean_up_tokenization_spaces=False, skip_special_tokens=True)+'\t'+tokenizer.decode(tgt_tensors[i].cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)+'\n')
    f.close()
    generator.train()
    return

def output(tensors, tokenizer, max_length=20, batch_size=64, generator=None, LM_pth='yelp.bin'):
    #tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    generator.eval()
    device=torch.device('cuda')
    dataset=Data.TensorDataset(tensors, tensors)
    data_iter=torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=False)
    output_tensors=[]
    output_operations=[]
    
    for tensor, _ in data_iter:
        tensor=remove_pad(tensor)
        tgt=torch.tensor([101 for _ in range(tensor.size(0))]).view(1,-1).to(device)
        output_tensors.append(tgt)
        for i in range(max_length-1):
            output=generator(tensor.t().to(device), tgt.to(device)).argmax(-1)
            tgt=torch.cat([tgt.clone().to(device), output.view(1,-1).clone().to(device)], 0).clone()
            output_tensors.append(output.view(1,-1))
        output_operations.append(torch.cat(output_tensors, 0).t())
        output_tensors=[]
    '''
    for tensor, _ in data_iter:
        output=generator(tensor.to(device)).argmax(-1)
        output_operations.append(output)
    '''
    '''
    output_operations=torch.cat(output_operations, 0)
    Operation=edit_operations(LM_pth)
    for i in range(tensors.size(0)):
        word_seq=Operation.seq_operate(tensors[i].to(device), output_operations[i].to(device))
        output_tensors.append(word_seq)
    '''
    return torch.cat(output_operations, 0)#output_tensors

def model_eval(generator, src, tgt, max_length=20):
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    generator.eval()
    device=torch.device('cuda')
    #decoder_input=torch.tensor([5 for _ in range(src.size(0))])
    #generated_sentence=generator(src.to(device)).argmax(-1)
    
    generated_sentence=[src.t().to(device)[0].view(1,-1)]
    for i in range(max_length-1):
        output=generator(src.t().to(device), tgt.t()[:i+1].to(device)).argmax(-1).view(-1)
        decoder=output.clone()
        generated_sentence.append(decoder.clone().view(1,-1))
    generated_sentence=torch.cat(generated_sentence, 0).t()
    
    for i in range(generated_sentence.size(0)):
        print('--------------------')
        print(tokenizer.decode(src[i].cpu().numpy().tolist(), skip_special_tokens=True))
        print(tokenizer.decode(generated_sentence[i].cpu().numpy().tolist(), skip_special_tokens=True))
        print(tokenizer.decode(tgt[i].cpu().numpy().tolist(), skip_special_tokens=True))
    generator.train()
    return
        
class Config():
    # training
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    BATCH_SIZE=32
    MAX_LENGTH=30
    classifier_pretrain_epoch=10
    classifier_threshold=0.95
    miu_a=0.8
    miu_b=0.2
    LM_n=5
    dim=768
    num_layers=4

    # evaluation
    attention_heads=8
    CNN_size='2,3,4'
    CNN_dim=768
    sentence_bert_type='bert-base-uncased'

def main():
    
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    device=torch.device('cuda')
    pth_pos='./Data/yelp/train.pos'
    pth_neg='./Data/yelp/train.neg'
    pth_pos_test='./Data/yelp/train.pos'
    pth_neg_test='./Data/yelp/train.neg'
    classifier_save_pth='./classifier.pth'
    LM_save_pth='./yelp_LM.pth'
    KENLM_pth='LM/yelp.bin'
    '''
    tokenizer=Tokenizer()
    tokenizer.build_vocab([pth_pos, pth_neg, pth_pos_test, pth_neg_test])
    '''
    BATCH_SIZE=32
    MAX_LENGTH=30
    classifier_pretrain_epoch=10
    print('Data Processing.')
    pos_tensors, neg_tensors=tensors(pth_pos, pth_neg, max_len=MAX_LENGTH)
    pos_tensors=torch.tensor(pos_tensors)
    neg_tensors=torch.tensor(neg_tensors)
 
    pos_tensors_test, neg_tensors_test=tensors(pth_pos_test, pth_neg_test, max_len=MAX_LENGTH)
    pos_tensors_test=torch.tensor(pos_tensors_test)
    neg_tensors_test=torch.tensor(neg_tensors_test)
    print('Classifier Pretraing.')
     
    classifier=pretrain_classifier(pos_tensors, neg_tensors, pos_tensors_test, neg_tensors_test, \
        vocab_size=tokenizer.vocab_size, epoch_num=classifier_pretrain_epoch, batch_size=BATCH_SIZE)
    torch.save(classifier.state_dict(), classifier_save_pth)
    
    classifier=TextMLP(tokenizer.vocab_size)
    classifier.load_state_dict(torch.load(classifier_save_pth))
    classifier=classifier.to(device)
    classifier.eval()
    
    generator=Transformer(words_number=tokenizer.vocab_size, layers=4)
    #generator=Bert()
    generator=generator.to(device)
    generator.train()
    
    train(tokenizer, pos_tensors_test, generator, classifier, lm_pth=LM_save_pth, kenlm_pth=KENLM_pth, epoch_num=20, max_length=MAX_LENGTH, batch_size=BATCH_SIZE, save_pth='./Generator.pth')
    return

if __name__=='__main__':
    main()
