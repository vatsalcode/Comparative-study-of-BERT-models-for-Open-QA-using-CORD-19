from COVIDModel import COVIDModel
import pickle
import os
import numpy as np
import torch
print('cos sci mean (abs + doc)')

def cosineSimilarity(vector1, vector2):
  #print(type(vector1), type(vector2))
  return np.dot(vector1,vector2)/(np.sqrt(np.dot(vector1,vector1)*np.dot(vector2,vector2)))

def euc(vector1, vector2):
  return np.absolute(np.dot(vector1,vector1) - np.dot(vector2,vector2))

def findCluster(input_embedding, cluster_embeddings):
  cosine_similarities = np.array([cosineSimilarity(input_embedding,cluster_embedding) for cluster_embedding in cluster_embeddings])
  return np.argmax(cosine_similarities)

def findDocs(input_embedding, cluster_embeddings):
  cosine_similarities = np.array([cosineSimilarity(input_embedding,cluster_embedding) for cluster_embedding in cluster_embeddings])
  print(np.sort(cosine_similarities))
  return np.argsort(cosine_similarities)

def findClusterEuc(input_embedding, cluster_embeddings):
  euclidean = np.array([euc(input_embedding,cluster_embedding) for cluster_embedding in cluster_embeddings])
  return np.argmax(euclidean)

def findDocsEuc(input_embedding, cluster_embeddings):
  euclidean = np.array([euc(input_embedding,cluster_embedding) for cluster_embedding in cluster_embeddings])
  print(np.sort(euclidean))
  return np.argsort(euclidean)


pool_types = ['mean']
bert_types = ['sci_uncased']
question_list = ["What do we know about vaccines and therapeutics ?",
                 "What do we know about virus genetics, origin, and evolution ?",
                 "What are the symptoms of an infected patient ?",
                 "How fatal is the coronavirus disease ?",
                 "What are the long term impacts of the virus ?",
                 "How long can the virus survive ?",
                 "What do we know about COVID genetics, origin and evolution ?",
                 "What are the ethical issues and concerns regarding Coronavirus ?",
                 "What do we know about the antibodies of coronavirus ?",
                 "How is COVID related to SARS ?",
                 "How does coronavirus affect pregnant women and neonates ?",
                 "How contagious is corornavirus ?",
                 "What are the recommended precautions to mitigate the viral disease ?",
                 "How does the disease affect smokers ?",
                 "How does the temperature affect the spread of the coronavirus ?"]
e_type = ['abstract_embedding', 'document_embedding']


for p in pool_types:
  if p=='mean':
    base_directory = 'new_embeddings_mean'
  elif p=='max':
    base_directory = 'new_embeddings'

  for bert in bert_types:
    if bert=='bio_cased':
      model_name = 'biobert'
    elif bert=='base_uncased':
      model_name = 'bert-base-uncased'
    elif bert=='sci_uncased':
      model_name = 'scibert-scivocab-uncased'

    cm = COVIDModel(model_name=model_name, max_length=500, stride=250)
    
    e_list = []
    for question in question_list:
      e = cm.getEmbedding(text=question, pooling_type=p)
      e = e.reshape(768)
      e = list(e.data.cpu().numpy())
      e_list.append(e)

    for embedding_type in e_type:
      if embedding_type=='abstract_embedding':
        n = 15
      else:
        n = 14
      
      cluster_id_to_doc_id = None
      
      c_to_d = "{}/{}/{}_cluster_id_to_doc_ids_{}.pkl".format(base_directory, bert, embedding_type, n)
      with open(c_to_d, "rb") as input_file:
        print(c_to_d)
        cluster_id_to_doc_id = pickle.load(input_file)

      cluster_id_to_centroid = None
      c_to_c = "{}/{}/{}_cluster_id_to_centroid_{}.pkl".format(base_directory, bert, embedding_type, n)
      with open(c_to_c, "rb") as input_file:
        cluster_id_to_centroid = pickle.load(input_file)

      list_of_centroids = cluster_id_to_centroid.values()
      print(len(list_of_centroids))
      for a in range(len(e_list)):
        e = e_list[a]
        cluster_id = findCluster(e, list_of_centroids)
        print("Cluster ID:{}".format(cluster_id))

        list_of_docs = cluster_id_to_doc_id[cluster_id]
        print(len(list_of_docs))
        # get embeddings of all docs from abstract embeddings
        doc_embeds = []
        doc_id_to_embed = {}
        directory = "{}/{}/{}".format(base_directory, bert, embedding_type)
        files = os.listdir(directory)

        for f in files:
          #print(f)
          if f in list_of_docs:
            file_name = directory+"/"+f
            embedding = torch.load(file_name)
            embedding = embedding.reshape(768)
            doc_embeds.append(list(embedding.data.cpu().numpy()))
            doc_id_to_embed[len(doc_embeds)-1] = f
        top_10_docs = []

        ids = findDocs(e, doc_embeds)
        loop_range = min(len(ids), 10)
        for i in range(loop_range):
          doc_index_sort = ids[len(ids)-1-i]
          top_10_docs.append(doc_id_to_embed[doc_index_sort])
        print(question_list[a])
        print(top_10_docs)
        print('------------------------------------------------')
      
      print('..................................')
    print('*****************************')
  print('######################################')
