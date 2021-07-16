from transformers import *
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
class COVIDModel:

    def __init__(self, model_name='bert-base-uncased', max_length=100, stride=50):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.max_length = max_length
        self.stride = stride
        if model_name == 'bert-base-uncased':
            configuration = BertConfig()
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel(configuration).from_pretrained(self.model_name)
            self.model.to(device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            #self.model.bert.embeddings.requires_grad = False
            self.embedding_size = 768
        if model_name == 'bert-large-uncased':
            configuration = BertConfig()
            self.embedding_size = 1024
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel(configuration).from_pretrained(self.model_name)
            self.model.to(device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            #self.model.bert.embeddings.requires_grad = False
        if model_name == 'scibert-scivocab-uncased':
            configuration = BertConfig()
            self.embedding_size = 768
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
            self.model.to(device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        if model_name == 'bert-base-multilingual-cased':
            configuration = BertConfig()
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel(configuration).from_pretrained(self.model_name)
            self.model.to(device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.embedding_size = 768
        if model_name == 'biobert':
            self.model  = BertModel.from_pretrained("biobert")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.model.to(device) 
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.embedding_size = 768
    def padTokens(self, tokens):
        if len(tokens)<self.max_length:
            tokens = tokens + ["[PAD]" for i in range(self.max_length - len(tokens))]
        return tokens

    def tokenizeText(self, tokens):
        tokens_array = []
        #window_movement_tokens =  max_length - stride
        for i in range(0, len(tokens), self.stride):
            if i+self.max_length<len(tokens):
                curr_tokens = ["[CLS]"] + tokens[i:i+self.max_length] + ["[SEP]"]
            else:
                padded_tokens = self.padTokens(tokens[i:i+self.max_length])
                curr_tokens = ["[CLS]"] + padded_tokens + ["[SEP]"]
            curr_tokens = self.tokenizer.convert_tokens_to_ids(curr_tokens)
            tokens_array.append(curr_tokens)
        return tokens_array
    def getEmbedding(self, text, if_pool=True, pooling_type="mean", batchsize = 1):
        tokens = self.tokenizer.tokenize(text)
        tokenized_array = self.tokenizeText(tokens)
        embeddingTensorsList = []
        print(len(tokenized_array))
        tensor = torch.zeros([1, 768], device=device)
        count = 0
        if len(tokenized_array)>batchsize:
            for i in range(0, len(tokenized_array), batchsize):
                current_tokens = tokenized_array[i:min(i+batchsize,len(tokenized_array))]
                token_ids = torch.tensor(current_tokens).to(device)
                seg_ids=[[0 for _ in range(len(tokenized_array[0]))] for _ in range(len(current_tokens))]
                seg_ids   = torch.tensor(seg_ids).to(device)
                hidden_reps, cls_head = self.model(token_ids, token_type_ids = seg_ids)
                cls_head.to(device)
                if if_pool and pooling_type=="mean":
                    tensor = tensor.add(torch.sum(cls_head, dim=0))
                    count +=cls_head.shape[0]
                elif if_pool and pooling_type == "max":
                    new_head = torch.max(cls_head, dim=0)[0]
                    tensor = torch.max(tensor, new_head)
                else:
                    embeddingTensorsList.append(cls_head)
                del cls_head, hidden_reps
            if if_pool and pooling_type=="mean" and count>0:
                embedding = torch.div(tensor, count)
            elif if_pool and pooling_type == "max":
                embedding = tensor
            elif not if_pool:
                embedding = torch.cat(embeddingTensorsList, dim=0)
            else:
                raise NotImplementedError()

        else:
            token_ids = torch.tensor(tokenized_array).to(device)
            seg_ids=[[0 for _ in range(len(tokenized_array[0]))] for _ in range(len(tokenized_array))]
            seg_ids   = torch.tensor(seg_ids).to(device)
            hidden_reps, cls_head = self.model(token_ids, token_type_ids = seg_ids)
            cls_head.to(device)
            cls_head.requires_grad = False
            if if_pool and pooling_type=="mean":
                embedding = torch.div(torch.sum(cls_head, dim=0), cls_head.shape[0])
            elif if_pool and pooling_type == "max":
                embedding = torch.max(cls_head, dim=0)[0]
            elif not if_pool:
                embedding = cls_head
            else:
                raise NotImplementedError()
            del cls_head, hidden_reps
        return embedding
