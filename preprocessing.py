import os, os.path, shutil
import json
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0

folder_paths = ["noncomm_use_subset/pdf_json", "custom_license/pdf_json", "comm_use_subset/pdf_json", "biorxiv_medrxiv/pdf_json"]
#folder_paths = ["noncomm_use_subset/pdf_json"]

def pre_process(text):

  try:
    t = text.encode('ascii', 'ignore')
    t = t.decode('utf8')
    l = detect(t)
    if l == 'en':
      return t
  except Exception as e:
    print('Error: {}'.format(text))

def check_num_of_words(t, num_of_words):
  if len(t.split(' ')) > num_of_words:
    return t

for f in folder_paths:
  pf = os.path.join("./processed_data/",f)
  if not os.path.exists(pf):
    os.makedirs(os.path.join("./processed_data/",f))
  try:
    num_files = os.listdir(f)
    print(len(num_files))
    for _file in num_files:
      try:
        full_filename = os.path.join(f, _file)
        if os.path.exists(full_filename):
          processed_data = {}
          with open(full_filename) as rf:
            data = json.load(rf)
            
            paper_id = data['paper_id']
            processed_data['paper_id'] = paper_id
            new_file = os.path.join('./processed_data', f, paper_id + '.json')
            if os.path.exists(new_file):
              continue
            # abstract
            abstract_list = data['abstract']
            abstract_new = []
            for i in abstract_list:
              j = i
              temp = i['text']
              temp = pre_process(temp)
              if temp:
                j['text'] = temp
                abstract_new.append(temp)
              
            abstract = check_num_of_words(" ".join(abstract_new), 10)
            if abstract:
              processed_data['abstract'] = abstract
            else:
              continue

            # body doc
            body_list = data['body_text']
            body_new = []
            for i in body_list:
              temp = i['text']
              temp = pre_process(temp)
              if temp:
                body_new.append(temp)
                
            full_text = check_num_of_words(' '.join(body_new), 100)
            if full_text:
              processed_data['body'] = body_new
            else:
              continue

          new_file = os.path.join('./processed_data', f, paper_id + '.json')
          with open(new_file, 'w+') as rf:
            json.dump(processed_data, rf, indent=4)
      except Exception as er:
        print(er, _file)
  except Exception as e:
    print(e, f) 




