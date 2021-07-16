from sklearn.cluster import DBSCAN 
import numpy as np
import os
import torch
import pickle

def createClusters(data, min_distance = 0.3, min_points_in_cluster = 10, doc_ids = []):
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

bert_types = ["new_embeddings_mean/sci_uncased"]
embedding_types = ["abstract_embedding", "document_embedding"]
#min_distance_types = [10, 5, 1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]
min_distance_types = np.logspace(1, -10, num=50)

for bert in bert_types:
  for embed in embedding_types:
    directory = os.path.join(bert, embed)
    doc_ids = []
    data = []
    doc_passage_ids = []
    embedding = None

    files = os.listdir(directory) # list of embedding files
    print(directory, len(files))

    for f in files:
      doc_ids.append(f) # doc id list
      embedding = torch.load(os.path.join(directory,f))
      embedding = embedding.reshape(768)
      data.append(list(embedding.data.cpu().numpy()))

    print("Loaded data:cos sci mean")
    print(len(data))

    for i_ in range(len(min_distance_types)):
      doc_info = {}
      cluster_info = {}
      cluster_id_to_doc_ids = {}
      labels, clusters, centroids = createClusters(data, min_distance = min_distance_types[i_], min_points_in_cluster = 10, doc_ids = doc_ids)
      doc_info = {doc_ids[i]:labels[i] for i in range(len(doc_ids))}
      cluster_info = {i:centroids[i] for i in range(len(centroids))}
      cluster_id_to_doc_ids = {i:clusters[i] for i in range(len(clusters))}
      
      print("eps {}, Number of clusters {}".format(min_distance_types[i_], len(clusters)))
      #for key, value in cluster_id_to_doc_ids.items():
      #  print('Cluster {} : {} documents'.format(key, len(value))) 
      with open('{}/{}_doc_to_cluster_{}.pkl'.format(bert, embed, str(i_)), 'wb') as f:
        pickle.dump(doc_info, f)
      with open('{}/{}_cluster_id_to_centroid_{}.pkl'.format(bert, embed, str(i_)), 'wb') as f:
        pickle.dump(cluster_info, f)
      with open('{}/{}_cluster_id_to_doc_ids_{}.pkl'.format(bert, embed, str(i_)), 'wb') as f:
        pickle.dump(cluster_id_to_doc_ids, f)

    print('Type {} is done'.format(embed))
    print('...................................')
  print('Type {} is done.'.format(bert))
  print('--------------------------------------')


