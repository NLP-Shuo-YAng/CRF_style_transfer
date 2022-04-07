import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertForSequenceClassification


class edit_operations():
    def __init__(self, LM_pth):
        tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        self.LM=BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=tokenizer.vocab_size)
        self.LM.load_state_dict(torch.load(LM_pth))
        self.LM.eval()
        self.device=torch.device('cuda')
        self.LM=self.LM.to(self.device)
    def _delete(self, tensor, position):
        return torch.cat([tensor[:position], tensor[position+1:]], -1)
    def _insert(self, tensor, position):
        inp=torch.cat([tensor[:position], torch.tensor([103]).to(self.device), tensor[position:]], -1)[:tensor.size(-1)]
        out=self.LM(inp.view(1,-1))[0].argmax(-1).view(-1)
        return torch.cat([tensor[:position], out, tensor[position:]], -1)[:self.max_len]
    def _substitute(self, tensor, position):
        inp=torch.cat([tensor[:position], torch.tensor([103]).to(self.device), tensor[position+1:]], -1)[:tensor.size(-1)]
        logits=self.LM(inp.view(1,-1))[0]
        out=logits.argmax(-1).view(-1)
        
        if int(out)==int(tensor[position]) and position<int(tensor.size(-1)):
            out=torch.tensor(logits.topk(2)[1].view(-1)[1].unsqueeze(0))
        
        return torch.cat([tensor[:position], out, tensor[position+1:]], -1)[:self.max_len]

    def seq_operate(self, tensor, operation):
        assert tensor.size(-1)==operation.size(-1)
        max_len=tensor.size(-1)
        tensor=tensor.view(-1).to(self.device)
        self.max_len=tensor.size(-1)
        operation_clean=[]
        flag=0
        for m in operation:
            if flag==0:
                if int(m)==6:
                    flag=1
                operation_clean.append(int(m))
            else:
                operation_clean.append(0)
        operation=torch.tensor(operation_clean).to(self.device)
        # 0-keep  1-insert  2-delete  3-substitute
        movement=0
        for i in range(tensor.size(-1)):
            '''
            print(operation)
            print(i)
            '''
            if int(operation[i])==1 and int(tensor[i])!=0:
                tensor=self._delete(tensor, i+movement)
                movement-=1
            elif int(operation[i]==2) and int(tensor[i])!=0:
                tensor=self._insert(tensor, i+movement)
                movement+=1
            elif int(operation[i]==3) and int(tensor[i])!=0:
                tensor=self._substitute(tensor, i+movement)
        tensor=torch.cat([tensor.view(1,-1), torch.tensor([[0 for _ in range(max_len-tensor.size(-1))]]).to(self.device)],-1)
        return tensor.view(1,-1)

        


