from sklearn.cluster import DBSCAN 
import numpy as np
import os
import torch
import pickle

def createClusters(data, min_distance = 0.3, min_points_in_cluster = 20, doc_ids = []):
  db = DBSCAN(eps=min_distance, min_samples=min_points_in_cluster, metric = 'cosine').fit(data)
  labels = db.labels_
  unique_labels = len(set(labels))

  clusters = [[] for i in range(unique_labels)]
  for label, datum in zip(labels,data):
    clusters[label].append(datum)

  centroids = []
  for cluster in clusters:
    centroids.append(np.mean(cluster,axis = 0))

  clusters = [[] for i in range(unique_labels)]
  for label, doc_id in zip(labels,doc_ids):
    clusters[label].append(doc_id)
  return labels, clusters, centroids

def cosineSimilarity(vector1, vector2):
  return np.dot(vector1,vector2)/(np.sqrt(np.dot(vector1,vector1)*np.dot(vector2,vector2)))

def findCluster(input_embedding, cluster_embeddings):
  cosine_similarities = np.array([cosineSimilarity(input_embedding,cluster_embedding) for cluster_embedding in cluster_embeddings])
  return np.argmax(cosine_similarities)

directory = "bert_embeddings_500_base_uncased/passage_embedding"
folders =  os.listdir(directory)
doc_ids = []
data = []
for folder in folders:
  path = directory + "/" + folder + "/pdf_json"
  files = os.listdir(path)
  print(path, len(files))
  for f in files:
    doc_ids.append(f)
    passages = os.listdir(os.path.join(path, f))
    for passage in passages:
      embedding = torch.load(os.path.join(path,f, passage))
      embedding = embedding.reshape(768)
      data.append(list(embedding.data.cpu().numpy()))
print("Loaded data")
print(len(data))

labels, clusters, centroids = createClusters(data, min_distance = 0.001, min_points_in_cluster = 40, doc_ids = doc_ids)
doc_info = {doc_ids[i]:labels[i] for i in range(len(doc_ids))}
cluster_info = {i:centroids[i] for i in range(len(centroids))}
cluster_id_to_doc_ids = {i:clusters[i] for i in range(len(clusters))}
prin://www.sec.gov/Archives/edgar/data/75288/000104746912003586/0001047469-12-003586-index.htmt(cluster_id_to_doc_ids)
print("Number of clusters", len(clusters))
with open('bert_embeddings_500_base_uncased_document_doc_info.pkl', 'wb') as f:
  pickle.dump(doc_info, f)

with open('bert_embeddings_500_base_uncased_document_cluster_id_to_centroid.pkl', 'wb') as f:
  pickle.dump(cluster_info, f)

with open('bert_embeddings_500_base_uncased_document_cluster_id_to_doc_ids.pkl', 'wb') as f:
  pickle.dump(cluster_id_to_doc_ids, f)
for key in cluster_id_to_doc_ids.keys():
 print("Cluster ID:{},Number of Documents:{}".format(key, len(cluster_id_to_doc_ids[key]) ))
