import nltk

class Tokenizer():
    def __init__(self):
        self.word2index={'[PAD]':0, '[MASK]':3, '[SEP]':102, '[CLS]':101}
        self.index2word={0:'[PAD]', 3:'[MASK]', 102:'[SEP]', 101:'[CLS]'}
        self.vocab_size=200
    def build_vocab(self, pth_list):
        print('Vocab Building.')
        lines=[]
        for pth in pth_list:
            f=open(pth)
            lines+=f.readlines()
            f.close()
        for line in lines:
            words_list=nltk.tokenize.word_tokenize(line)
            for w in words_list:
                if w not in self.word2index:
                    self.word2index[w]=self.vocab_size
                    self.index2word[self.vocab_size]=w
                    self.vocab_size+=1
        print('Finish building.')
        return
    def encode(self, sen):
        if type(sen)==list:
            sen=' '.join(sen)
        words_list=nltk.tokenize.word_tokenize(sen)
        index_list=[101]
        for w in words_list:
            if w not in words_list:
                print('Warning: find an uncommon word.')
                pass
            else:
                index_list.append(self.word2index[w])
        index_list.append(102)
        return index_list
    def decode(self, index, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        special_tokens=[0, 102, 101]
        sen=[]
        for i in index:
            if skip_special_tokens:
                if i in special_tokens:
                    pass
                else:
                    sen.append(self.index2word[i])
            else:
                sen.append(self.index2word[i])
        if clean_up_tokenization_spaces:
            return ' '.join(sen)
        else:
            return ' '.join(sen)
    def tokenize(self, sen):
        return nltk.tokenize.word_tokenize(sen)

