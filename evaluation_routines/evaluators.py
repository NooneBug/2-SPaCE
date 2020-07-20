from sklearn.metrics import pairwise_distances
import numpy as np
from numpy.linalg import norm

def hyper_distance(x, y):
  return np.arccosh(1 + ((2 * (norm(x - y) ** 2)) / ((1 - norm(x) ** 2)*(1 - norm(y) ** 2))))

def cosine_similarity(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))

def cosine_dissimilarity(a, b):
    return 1 - cosine_similarity(a, b)

class RegressionEvaluator():

  def __init__(self, predictions, labels, type_lookup):

    self.predictions = predictions
    self.labels = labels
    self.types = {k: v.model.weight.detach().cpu().numpy() for k, v in type_lookup.items()}
    self.type_lookups = type_lookup
    self.names = list(type_lookup.keys())

    self.pairwise_distances = {'type2vec': cosine_dissimilarity, 'nickel_hyperbolic': hyper_distance}


  def evaluate(self, folder_name):
    for name in self.names:
      pred = self.predictions[name]
      labels = self.labels
      type_vectors = self.types[name]

      # print('---- {} ----'.format(name))
      # print('pred: {}'.format(pred))
      # print('labels: {}'.format(labels))		
      # print('types: {}'.format(types))

      distances = pairwise_distances(pred, type_vectors, self.pairwise_distances[name])

      print('distances: {}'.format(distances))

      with open(folder_name + '/result_file.txt', 'a') as out, open(folder_name + '/parsable_result_file.txt', 'a') as parsable_out:
        out.write('results for {}:\n'.format(name))

        all_types = ' '.join(self.type_lookups[name].emb.get_ordered_typelist())

        out.write('\ttypes: {}\n'.format(all_types))

        parsable_out.write('{}|{}\n'.format(name, all_types))

        for i, (distance, example_labels) in enumerate(zip(distances, labels)):
          out.write('\t ------ {}th example ------\n'.format(i))
          out.write('\t\ttrue labels: {}\n'.format(example_labels)) 
          out.write('\t\tdistances: {}\n'.format(str(distance)))
          sorted_list = sorted(distance)
          type_ranking = ' '.join([str(sorted_list.index(d)) for d in distance])
          out.write('\t\tdistance_order: {}\n'.format(type_ranking))
          parsable_out.write('\t{}|{}|{}|{}\n'.format(i, example_labels, str(distance), type_ranking))

      out.close()
      parsable_out.close()      

class ClassifierEvaluator():

  def __init__(self, predictions, labels, type_lookup):
    self.predictions = predictions
    self.labels = labels
    
    self.all_types = list(type_lookup.values())[0].emb.get_ordered_typelist()
    # self.type_lookups = type_lookup

  def evaluate(self, folder_name):
    with open(folder_name + '/result_file.txt', 'a') as out, open(folder_name + '/parsable_result_file.txt', 'a') as parsable_out:
      out.write('result for classification:\n')
      out.write('\tall types: {}\n'.format(self.all_types))

      parsable_out.write('classification | {}\n'.format(self.all_types))

      print('predictions: {}'.format(self.predictions))

      for i, (prediction, example_labels) in enumerate(zip(self.predictions, self.labels)):
        out.write('\t ------ {}th example ------\n'.format(i))
        out.write('\t\ttrue labels: {}\n'.format(example_labels)) 
        out.write('\t\tpredictions: {}\n'.format(prediction))

        parsable_out.write('\t{}|{}|{}\n'.format(i, example_labels, prediction))
    out.close()
    parsable_out.close()

    
