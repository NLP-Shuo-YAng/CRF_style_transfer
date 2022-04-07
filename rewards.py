import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import kenlm
from transformers import BertTokenizer

class Rewards():
    def __init__(self, classifier, kenlm_pth='LM/yelp_neg.bin', belta=0.05, delta=0.01):#1.8343):
        self.device=torch.device('cuda')
        self.classifier=classifier#.to(self.device)
        self.LM=kenlm.LanguageModel(kenlm_pth)
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        self.smooth = SmoothingFunction()
        self.belta=belta
        self.delta=delta
    def style_rewards(self, sentence, reference):
        #print(sentence.size())
        rewards=F.softmax(self.classifier(sentence.view(1,-1)), -1).view(-1)
        src_rewards=F.softmax(self.classifier(reference.view(1, -1).to(self.device)), -1).view(-1)
        return float(rewards[1]-src_rewards[1])
    def content_rewards(self, sentence, reference):
        reference=[reference.view(-1).cpu().numpy().tolist()]
        sentence=sentence.view(-1).cpu().numpy().tolist()
        rewards=sentence_bleu(reference, sentence, smoothing_function=self.smooth.method1)
        return float(rewards)
    def lm_rewards(self, sentence, reference):
        lm_reward=self.LM.score(self.tokenizer.decode(sentence.view(-1).cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False))
        src_reward=self.LM.score(self.tokenizer.decode(reference.view(-1).cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False))
        return float(lm_reward)-float(src_reward)

    def total_rewards(self, sentence, reference):
        reward=self.style_rewards(sentence, reference)+\
               self.delta*(self.lm_rewards(sentence, reference))
        return reward

class policy_gradient_loss(nn.Module):
    def __init__(self):
        super(policy_gradient_loss, self).__init__()
        self.nll=nn.NLLLoss()
        self.device=torch.device('cuda')
    def forward(self, rewards, action, pro):
        #action=action[0]
        one_hot = torch.zeros(pro.size(0),4).to(self.device).scatter_(1,action.view(-1,1).to(self.device),1)
        rewards_metric=(one_hot.t()*rewards).t()
        log_pro=torch.log(pro)
        Loss=self.nll(log_pro*rewards_metric, action.view(-1))
        return Loss
