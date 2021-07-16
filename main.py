import torch
from transformers import BertTokenizer, BertModel, BertConfig
import os, os.path, shutil
import json
import pickle
from COVIDModel import COVIDModel

# loop through all files to :
# 1. fetch abstract and paper id.
# 2. get embeddings for all windows for the abstract.
# 3. pool them (default = mean).
# 3. store pooled embeddings in a pickle file by name as paper_id.
base_dir = "processed_data"
#folder_paths = ["noncomm_use_subset/pdf_json", "custom_license/pdf_json", "comm_use_subset/pdf_json", "biorxiv_medrxiv/pdf_json"]
folder_paths = ["custom_license/pdf_json"]
new_folder = "new_embeddings1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cm = COVIDModel(model_name='bert-base-uncased', max_length=500, stride=250)

for f in folder_paths:
  #list_of_id_and_embedding = []
  af = os.path.join(new_folder, "base_uncased/abstract_embedding/")
  pf = os.path.join(new_folder, "base_uncased/passage_embedding/")
  df = os.path.join(new_folder, "base_uncased/document_embedding/")
  if not os.path.exists(af):
    os.makedirs(af)
  if not os.path.exists(df):
    os.makedirs(df)
  if not os.path.exists(pf):
    os.makedirs(pf)
  try:
    num_files = os.listdir(os.path.join(base_dir,f))
    print(len(num_files))
    for _file in num_files:
      try:
        full_filename = os.path.join(base_dir,f, _file)
        if os.path.exists(full_filename):
          with open(full_filename) as rf:
            print('a')
            data = json.load(rf)
            print(data.keys())
            paper_id = data['paper_id']
            print('b')
            abstract = data['abstract']
            abstract_temp = os.path.join(af, paper_id + '.pt')
            if not os.path.exists(abstract_temp):
              embeddings = cm.getEmbedding(text=abstract, pooling_type="max")
              torch.save(embeddings, abstract_temp)
            print('c')
            body_list = data['body']
            full_text = ' '.join(body_list)
            full_text = abstract + ' ' + full_text
            for i in range(len(body_list)):
              passage_temp = os.path.join(pf, paper_id, paper_id + '_' + str(i) + '.pt')
              if not os.path.exists(passage_temp):
                passage_folder = os.path.join(pf, paper_id)
                if not os.path.exists(passage_folder):
                  os.makedirs(passage_folder)
                embeddings = cm.getEmbedding(text=body_list[i], pooling_type="max")
                torch.save(embeddings, passage_temp)
            doc_temp = os.path.join(df, paper_id + '.pt')
            if not os.path.exists(doc_temp):
              embeddings = cm.getEmbedding(text=full_text, pooling_type="max")
              torch.save(embeddings, doc_temp)
        else:
          print(full_filename + ' doesn\'t exist.')
      except Exception as er:
        print(er, _file)
        break
  except Exception as e:
    print(e, f)
    break
