# Script Showing how to Run the Model 
# as well as printing the event embeddings using the Model

import tensorflow as tf
import numpy as np
from glove_utils import Glove
import argparse
from dataset import SVO, ContextualOpenIE_Dataset
import itertools
import math
import string
from scipy import spatial
from scipy import stats

##################
# Use this dataset if each line contains a svo triple in the form
# subject|verb|object|sentence
# Where sentence is the sentence it appears in (may be blank if you dont need it)
##################
class SimpleDataset:  
    'regular svo triple input format'
    def __init__(self, filename):
        self.filename=filename

    def __iter__(self):  #generates a svo triple along with a sentence it appears in
        with open(self.filename, 'r') as fi:
            for line in fi:
                print(line)
                splits = line.split('|')
                print(splits)
                yield(SVO(splits[0], splits[1], splits[2], splits[3]))
        return None #return None for the last item


common_verbs = ['be', 'were', 'been', 'is', "'s", 'have', 'had', 'do', 'did', 'done', 'say', 'said', 'go', 'went', 'gone', 'get', 'got', 'gotton']
def get_average(embed_list, embeddings):
    n = len(embed_list)
    if n == 0:
        return None
    dim = embeddings.shape[1]
    avg = np.zeros(dim)
    for i in embed_list:
        avg += embeddings[i, :] 
    return avg / n

def run_model(svo_file, embeddings, out):
    embed_dim=embeddings.dim

    with tf.Session() as sess:
#        saver = tf.train.import_meta_graph('checkpoints/model.ckt.meta')
#        saver.restore(sess, 'checkpoints/model.ckt')
        saver = tf.train.import_meta_graph('{}/{}.meta'.format(FLAGS.checkpoint_dir, FLAGS.model_name))
        saver.restore(sess, '{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.model_name))

        graph = tf.get_default_graph()

        subject_ph_in = tf.placeholder(tf.float32, shape=(1,embed_dim))
        verb_ph_in = tf.placeholder(tf.float32, shape=(1,embed_dim))
        object_ph_in = tf.placeholder(tf.float32, shape=(1,embed_dim))
        
        #Rebuild the graph
        if FLAGS.pred:
            print("Using Predicate Tensor Model")
            W = graph.get_tensor_by_name("W:0")
            U = graph.get_tensor_by_name("U:0")
            e = graph.get_tensor_by_name("Embed:0")
            alpha = tf.einsum('ijk,ci->jkc', W, verb_ph_in)
            obs = tf.stack([tf.diag(x) for x in tf.unstack(object_ph_in, axis=0)], axis=2)
            gamma = tf.einsum('iak,ajk->ijk', obs, alpha)
            L = tf.einsum('ijk,jkc->ikc', U, gamma)
            final = tf.einsum('jic,cj->ci', L, subject_ph_in) #ROWS are the embeddings for the batch

        else: #Role Factored
            print(" Using Role Factored Model")
            W = graph.get_tensor_by_name("W:0")
            e = graph.get_tensor_by_name("Embed:0")
            tensor= graph.get_tensor_by_name("tensor:0")
            vs = tf.einsum('ijk,ck,cj->ci', tensor,subject_ph_in,verb_ph_in)
            os = tf.einsum('ijk,ck,cj->ci', tensor,object_ph_in,verb_ph_in)
            svo = tf.concat([vs,os], 1) 
            final = tf.matmul(svo, W)


        ###Run the model through the data and output the embeddings to a text file

        data=ContextualOpenIE_Dataset(svo_file) 
 #       data=SimpleDataset(svo_file)  #Uncomment if you want to use a SimpleDataset

        outfile = open(out, 'w')
        embed = e.eval()
        i=0
        for d in data:
            subj = d.subject
            obj = d.obj
            verb = d.verb

            subj = embeddings.transform(subj.lower().split())
            verb= embeddings.transform(verb.lower().split())
            obj= embeddings.transform(obj.lower().split())

            subj = get_average(subj, embed)
            verb= get_average(verb, embed)
            obj= get_average(obj, embed)

            if d.verb not in common_verbs and subj is not None and verb is not None and obj is not None:

                feed_dict = {
                  subject_ph_in: np.array([subj]),
                  verb_ph_in: np.array([verb]),
                  object_ph_in: np.array([obj])
                }

                vec = sess.run(final, feed_dict=feed_dict)
                outfile.write("{}|{}|{}|{}\n".format(d.subject, d.verb, d.obj, ",".join(vec[0].astype(str))))
                print("Processed {}".format(i))
                i+=1
                if i >= FLAGS.num_embeddings:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', type=str, default='../data/glove.6B.100d.txt', help='File containing pre trained Glove word embeddings')
    parser.add_argument('--svo_file', type=str, default='svo_tuples_data.txt', help='A dataset file in the same form of that you trained with') 
    parser.add_argument('--out_file', type=str, default='output.txt', help='Where to output the resulting event embeddings')

    parser.add_argument('--pred', action="store_true") #Use the Predicate Tensor Model, else default to the Role Factor Model
    parser.add_argument('--num_embeddings', type=int, default=10000, help='How many embeddings to print out to the output file')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Where the pretrained model checkpoints are')
    parser.add_argument('--model_name', type=str, default='model.ckt', help='Model name')

    FLAGS = parser.parse_args()
    embeddings=Glove(FLAGS.embedding_file)
    embed_dim = embeddings.dim
    
    run_model(FLAGS.svo_file, embeddings, FLAGS.out_file)
    

