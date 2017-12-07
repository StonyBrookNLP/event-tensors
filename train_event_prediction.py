##################################################
# Train the event predication objective
# Contains code for the predicate tensor, role factor, and 
# comp neural network
#
# Try to minimize the cosine distance between 
# representations of events in the same discourse
#
# Unfortunatly, much of the code here is very similar
# to the code defined in train_word_prediction.py, so 
# if something is not clear here, I would check there
##################################################
import tensorflow as tf
import numpy as np
import utils.dataset as dataset
from utils.glove_utils import Glove
from utils.train_utils import EventPredQueuedInstances
import argparse
import itertools
import math
import string
from functools import reduce
from collections import deque

FLAGS = None

def placeholder_inputs(embeddings):
    """Embedding Input Placeholders"""
    embed_dim = embeddings.vocab_size

    subject_ph = tf.placeholder(tf.int64, shape=(FLAGS.batch_size * FLAGS.max_phrase_size), name="a")
    verb_ph = tf.placeholder(tf.int64, shape=[FLAGS.batch_size * FLAGS.max_phrase_size], name="b")
    object_ph = tf.placeholder(tf.int64, shape=[FLAGS.batch_size * FLAGS.max_phrase_size], name="c")

    subject_w = tf.placeholder(tf.float32, shape=(FLAGS.batch_size * FLAGS.max_phrase_size), name="d")
    verb_w = tf.placeholder(tf.float32, shape=(FLAGS.batch_size * FLAGS.max_phrase_size), name="e")
    object_w = tf.placeholder(tf.float32, shape=(FLAGS.batch_size * FLAGS.max_phrase_size), name="f")

    return subject_ph, verb_ph, object_ph, subject_w, verb_w, object_w


def fill_feed_dict(instance_iter, input_ph, target_ph, neg_ph, embeddings):
    """
    EventPredQueuedInstances instance_iter
    InputTargetIterator in
    neg_instances - InputTargetIterator object for negative instances
    input_ph, target_ph, neg_ph - each are a list of placeholder objects (returned from placeholder_inputs)
    embeddings should be a Glove embeddings object
    Return the feed dict as well as a flag indicating if we are finished
    """

    shape = np.array([FLAGS.batch_size, embeddings.vocab_size])
    embed_dim = embeddings.dim

    input_subject_ph = input_ph[0]
    input_verb_ph = input_ph[1]
    input_object_ph = input_ph[2]
    input_subject_w_ph = input_ph[3]
    input_verb_w_ph = input_ph[4]
    input_object_w_ph = input_ph[5]

    target_subject_ph =target_ph[0]
    target_verb_ph =target_ph[1]
    target_object_ph =target_ph[2]
    target_subject_w_ph =target_ph[3]
    target_verb_w_ph =target_ph[4]
    target_object_w_ph =target_ph[5]

    neg_subject_ph =neg_ph[0]
    neg_verb_ph =neg_ph[1]
    neg_object_ph =neg_ph[2]
    neg_subject_w_ph =neg_ph[3]
    neg_verb_w_ph =neg_ph[4]
    neg_object_w_ph =neg_ph[5]

    input_sub_id_values = []
    input_sub_weight_values = []
    input_verb_id_values = []
    input_verb_weight_values = []
    input_obj_id_values = []
    input_obj_weight_values = []

    target_sub_id_values = []
    target_sub_weight_values = []
    target_verb_id_values = []
    target_verb_weight_values = []
    target_obj_id_values = []
    target_obj_weight_values = []
    
    neg_sub_id_values = []
    neg_sub_weight_values = []
    neg_verb_id_values = []
    neg_verb_weight_values = []
    neg_obj_id_values = []
    neg_obj_weight_values = []

    done = False
    for i in range(FLAGS.batch_size):
        inst = next(instance_iter)
        input_inst = inst[0]
        target_inst = inst[1]
        neg_inst = inst[2]
        #print("Input: {}, Target: {}".format(inst[2], inst[3]))
        if input_inst and target_inst and neg_inst:

            input_sub_id, input_sub_w = input_inst[0]
            input_verb_id, input_verb_w = input_inst[1]
            input_obj_id, input_obj_w = input_inst[2]

            target_sub_id, target_sub_w = target_inst[0]
            target_verb_id, target_verb_w = target_inst[1]
            target_obj_id, target_obj_w = target_inst[2]
           
            neg_sub_id, neg_sub_w = neg_inst[0]
            neg_verb_id, neg_verb_w = neg_inst[1]
            neg_obj_id, neg_obj_w = neg_inst[2]

            input_sub_id_values.extend(input_sub_id)
            input_sub_weight_values.extend(input_sub_w)
            input_verb_id_values.extend(input_verb_id)
            input_verb_weight_values.extend(input_verb_w)
            input_obj_id_values.extend(input_obj_id)
            input_obj_weight_values.extend(input_obj_w)

            target_sub_id_values.extend(target_sub_id)
            target_sub_weight_values.extend(target_sub_w)
            target_verb_id_values.extend(target_verb_id)
            target_verb_weight_values.extend(target_verb_w)
            target_obj_id_values.extend(target_obj_id)
            target_obj_weight_values.extend(target_obj_w)

            neg_sub_id_values.extend(neg_sub_id)
            neg_sub_weight_values.extend(neg_sub_w)
            neg_verb_id_values.extend(neg_verb_id)
            neg_verb_weight_values.extend(neg_verb_w)
            neg_obj_id_values.extend(neg_obj_id)
            neg_obj_weight_values.extend(neg_obj_w)

        else: #reached the end of instances
            done = True
            break

    input_sub_weight_values = np.array(input_sub_weight_values)
    input_verb_weight_values = np.array(input_verb_weight_values)
    input_obj_weight_values = np.array(input_obj_weight_values)
    input_sub_id_values = np.array(input_sub_id_values)
    input_verb_id_values = np.array(input_verb_id_values)
    input_obj_id_values = np.array(input_obj_id_values)

    target_sub_weight_values = np.array(target_sub_weight_values)
    target_verb_weight_values = np.array(target_verb_weight_values)
    target_obj_weight_values = np.array(target_obj_weight_values)
    target_sub_id_values = np.array(target_sub_id_values)
    target_verb_id_values = np.array(target_verb_id_values)
    target_obj_id_values = np.array(target_obj_id_values)

    neg_sub_weight_values = np.array(neg_sub_weight_values)
    neg_verb_weight_values = np.array(neg_verb_weight_values)
    neg_obj_weight_values = np.array(neg_obj_weight_values)
    neg_sub_id_values = np.array(neg_sub_id_values)
    neg_verb_id_values = np.array(neg_verb_id_values)
    neg_obj_id_values = np.array(neg_obj_id_values)

    feed_dict = {
      input_subject_ph: input_sub_id_values,
      input_verb_ph: input_verb_id_values,
      input_object_ph: input_obj_id_values,
      input_subject_w_ph: input_sub_weight_values,
      input_verb_w_ph: input_verb_weight_values,
      input_object_w_ph: input_obj_weight_values,

      target_subject_ph: target_sub_id_values,
      target_verb_ph: target_verb_id_values,
      target_object_ph: target_obj_id_values,
      target_subject_w_ph: target_sub_weight_values,
      target_verb_w_ph: target_verb_weight_values,
      target_object_w_ph: target_obj_weight_values,

      neg_subject_ph: neg_sub_id_values,
      neg_verb_ph: neg_verb_id_values,
      neg_object_ph: neg_obj_id_values,
      neg_subject_w_ph: neg_sub_weight_values,
      neg_verb_w_ph: neg_verb_weight_values,
      neg_object_w_ph: neg_obj_weight_values
    }
    return feed_dict,done

def additive_nn(placeholder, embeddings, indices):
    """
    Compositional Neural Network Model
    """ 
    subject_ph = placeholder[0]
    verb_ph = placeholder[1]
    object_ph = placeholder[2]
    subject_ph_w = placeholder[3]
    verb_ph_w = placeholder[4]
    object_ph_w = placeholder[5]

    #The main network, compute the svo representation
    embed_dim = embeddings.dim 
    embd = embeddings.embd.astype(np.float32)

    embed_layer = tf.get_variable("Embed", shape=[embeddings.vocab_size,embed_dim], initializer=tf.constant_initializer(value=embd, verify_shape=True))
    W = tf.get_variable("W", shape=[3*embed_dim, FLAGS.hidden_size], initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embed_dim)))
    H = tf.get_variable("H", shape=[FLAGS.hidden_size,embed_dim], initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embed_dim)))

    subject_indices = tf.SparseTensor(indices, subject_ph, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    verb_indices = tf.SparseTensor(indices, verb_ph, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    object_indices = tf.SparseTensor(indices, object_ph, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    subject_weights = tf.SparseTensor(indices, subject_ph_w, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    verb_weights = tf.SparseTensor(indices, verb_ph_w, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    object_weights = tf.SparseTensor(indices, object_ph_w, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))

    print(subject_indices.get_shape())
    print(subject_weights.get_shape())

    subject_avg = tf.nn.embedding_lookup_sparse(embed_layer, subject_indices, subject_weights, combiner='sum')
    verb_avg = tf.nn.embedding_lookup_sparse(embed_layer, verb_indices, verb_weights, combiner='sum')
    object_avg = tf.nn.embedding_lookup_sparse(embed_layer, object_indices, object_weights, combiner='sum')

    verb_foo = tf.reshape(verb_avg, shape=[FLAGS.batch_size, embed_dim])
    subject_foo = tf.reshape(subject_avg, shape=[FLAGS.batch_size, embed_dim])
    object_foo = tf.reshape(object_avg, shape=[FLAGS.batch_size, embed_dim])

    svo = tf.concat([verb_foo, subject_foo, object_foo], 1) #need to make it batchsizeXdim for multiplication
    hidden = tf.tanh(tf.matmul(svo, W)) #hidden layer, no biases
    final = tf.matmul(hidden, H, name='final')
    return final, W, H

def role_factor_network(placeholder, embeddings, indices):
    """
    Role Factored Network
    """ 

    subject_ph = placeholder[0]
    verb_ph = placeholder[1]
    object_ph = placeholder[2]
    subject_ph_w = placeholder[3]
    verb_ph_w = placeholder[4]
    object_ph_w = placeholder[5]


    embed_dim = embeddings.dim 
    embd = embeddings.embd.astype(np.float32)

    tensor = tf.get_variable("tensor", shape=[embed_dim, embed_dim, embed_dim], initializer=tf.truncated_normal_initializer(stddev=1.0 / embed_dim))
    embed_layer = tf.get_variable("Embed", shape=[embeddings.vocab_size,embed_dim], initializer=tf.constant_initializer(value=embd, verify_shape=True))
    W = tf.get_variable("W", shape=[2*embed_dim, embed_dim], initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(embed_dim)))

    subject_indices = tf.SparseTensor(indices, subject_ph, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    verb_indices = tf.SparseTensor(indices, verb_ph, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    object_indices = tf.SparseTensor(indices, object_ph, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    subject_weights = tf.SparseTensor(indices, subject_ph_w, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    verb_weights = tf.SparseTensor(indices, verb_ph_w, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    object_weights = tf.SparseTensor(indices, object_ph_w, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))

    print(subject_indices.get_shape())
    print(subject_weights.get_shape())

    subject_avg = tf.nn.embedding_lookup_sparse(embed_layer, subject_indices, subject_weights, combiner='sum')
    verb_avg = tf.nn.embedding_lookup_sparse(embed_layer, verb_indices, verb_weights, combiner='sum')
    object_avg = tf.nn.embedding_lookup_sparse(embed_layer, object_indices, object_weights, combiner='sum')
    
    verb_foo = tf.reshape(verb_avg, shape=[FLAGS.batch_size, embed_dim])
    subject_foo = tf.reshape(subject_avg, shape=[FLAGS.batch_size, embed_dim])
    object_foo = tf.reshape(object_avg, shape=[FLAGS.batch_size, embed_dim])

    vs = tf.einsum('ijk,ck,cj->ci', tensor, subject_foo, verb_foo)
    vo = tf.einsum('ijk,ck,cj->ci', tensor, object_foo, verb_foo)

    svo = tf.concat([vs,vo], 1) 
    final = tf.matmul(svo, W, name='final')
    
    return final, tensor, W

def predicate_tensor_network(placeholder, embeddings, indices):
    """
    The predicate tensor network
    """ 
    #The main network, compute the svo representation
    
    subject_ph = placeholder[0]
    verb_ph = placeholder[1]
    object_ph = placeholder[2]
    subject_ph_w = placeholder[3]
    verb_ph_w = placeholder[4]
    object_ph_w = placeholder[5]

    embed_dim = embeddings.dim 
    embd = embeddings.embd.astype(np.float32)
    embed_layer = tf.get_variable("Embed", shape=[embeddings.vocab_size,embed_dim], initializer=tf.constant_initializer(value=embd, verify_shape=True))
    W = tf.get_variable("W", shape=[embed_dim, embed_dim, embed_dim], initializer=tf.truncated_normal_initializer(stddev=1.0 / embed_dim))
    U = tf.get_variable("U", shape=[embed_dim, embed_dim, embed_dim], initializer=tf.truncated_normal_initializer(stddev=1.0 / embed_dim))

    subject_indices = tf.SparseTensor(indices, subject_ph, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    verb_indices = tf.SparseTensor(indices, verb_ph, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    object_indices = tf.SparseTensor(indices, object_ph, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    subject_weights = tf.SparseTensor(indices, subject_ph_w, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    verb_weights = tf.SparseTensor(indices, verb_ph_w, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    object_weights = tf.SparseTensor(indices, object_ph_w, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))

    subject_avg = tf.nn.embedding_lookup_sparse(embed_layer, subject_indices, subject_weights, combiner='sum')
    verb_avg = tf.nn.embedding_lookup_sparse(embed_layer, verb_indices, verb_weights, combiner='sum')
    object_avg = tf.nn.embedding_lookup_sparse(embed_layer, object_indices, object_weights, combiner='sum')

    verb_foo = tf.reshape(verb_avg, shape=[FLAGS.batch_size, embed_dim])
    subject_foo = tf.reshape(subject_avg, shape=[FLAGS.batch_size, embed_dim])
    object_foo = tf.reshape(object_avg, shape=[FLAGS.batch_size, embed_dim])

    alpha = tf.einsum('ijk,ci->jkc', W, verb_foo)
    obs = tf.stack([tf.diag(x) for x in tf.unstack(object_foo, axis=0)], axis=2)
    gamma = tf.einsum('iak,ajk->ijk', obs, alpha)
    L = tf.einsum('ijk,jkc->ikc', U, gamma)

    final = tf.einsum('jic,cj->ci', L, subject_foo) #ROWS are the embeddings for the batch

    return final, W, U


def prediction_network(input_ph, target_ph, neg_ph, embeddings):
    """
    Given two events in the same discourse, get their representations (using the choosen architecture), and calculate the cosine similarity loss objective
    Note that input_ph, target_ph and neg_ph are all lists of placeholders for the input, target, and negative sample
    """
    indices = np.array([[b,x] for b in range(FLAGS.batch_size) for x in range(FLAGS.max_phrase_size)], dtype=np.int64)

    with tf.variable_scope("Network") as scope:
        print("Using get_embedding()")
        if FLAGS.role_factor:
            input_embed, tensor,W= role_factor_network(input_ph, embeddings, indices)  #tensor where rows are embeddings
            scope.reuse_variables()
            target_embed, _, _= role_factor_network(target_ph, embeddings, indices)
            neg_embed, _, _ = role_factor_network(neg_ph, embeddings, indices) #negative sample
        elif FLAGS.predicate_tensor:
            input_embed, tensor,W=predicate_tensor_network(input_ph, embeddings, indices)  #tensor where rows are embeddings
            scope.reuse_variables()
            target_embed, _, _=predicate_tensor_network(target_ph, embeddings, indices)
            neg_embed, _, _ =predicate_tensor_network(neg_ph, embeddings, indices) #negative sample
        else: #comp neural network
            input_embed, tensor,W=additive_nn(input_ph, embeddings, indices)  #tensor where rows are embeddings
            scope.reuse_variables()
            target_embed, _, _=additive_nn(target_ph, embeddings, indices)
            neg_embed, _, _ =additive_nn(neg_ph, embeddings, indices) #negative sample

    input_norm = tf.norm(input_embed, axis=1)
    target_norm= tf.norm(target_embed, axis=1)
    neg_norm= tf.norm(neg_embed, axis=1)

    #need to normalize first 
    input_unit = input_embed / tf.reshape(input_norm, shape=[FLAGS.batch_size, 1])
    target_unit = target_embed / tf.reshape(target_norm,shape=[FLAGS.batch_size,1])
    neg_unit = neg_embed / tf.reshape(neg_norm, shape=[FLAGS.batch_size,1])
    #compute the cosine similarities
    input_target_cos = 1-tf.losses.cosine_distance(input_unit, target_unit, dim=1, reduction=tf.losses.Reduction.NONE)
    input_neg_cos = 1-tf.losses.cosine_distance(input_unit, neg_unit, dim=1, reduction=tf.losses.Reduction.NONE)

    diff = input_target_cos - input_neg_cos #want this difference to be high
    #hinge loss
    loss = tf.maximum(0.0, FLAGS.margin - diff) + FLAGS.reg_lambda*tf.nn.l2_loss(W) + FLAGS.reg_lambda*tf.nn.l2_loss(tensor)

    mean = tf.reduce_mean(loss, name="mean_loss")
    return mean

def train_prediction_network(instances, embeddings):
    """
    EventPredQueuedInstances instances
    Glove embeddings
    """
    with tf.variable_scope("input"):
        input_ph = placeholder_inputs(embeddings)
    with tf.variable_scope("target"):
        target_ph = placeholder_inputs(embeddings)
    with tf.variable_scope("neg"):
        neg_ph = placeholder_inputs(embeddings)

    inst_iter=iter(instances)

    loss = prediction_network(input_ph, target_ph, neg_ph, embeddings)

    optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #Cry havoc! and let slip the dogs of war!
        if FLAGS.resume:
            print("Restoring Model from {}".format(FLAGS.restore_point))
            saver.restore(sess, FLAGS.restore_point)
            print("Model Restored")
        else:
            print("Starting Fresh")
            init = tf.global_variables_initializer()
            sess.run(init)


        done = False
        i = 0
        avg_loss=0
        avg_loss2=0
        while not done:
            feed_dict, done = fill_feed_dict(inst_iter, input_ph, target_ph, neg_ph, embeddings)
            _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
            i+=1
            avg_loss += (loss_val/200)
            avg_loss2 +=(loss_val/10000)
#            print(loss_val)
            
            if i % 200 == 0:
                print("Average Loss on {} is {}".format(i, avg_loss))
                avg_loss =0
            if i % 10000 == 0:
                print("Average Loss for past 10000 is {}".format(avg_loss2))
                avg_loss2 = 0

                print("Checkpoint Saved to {}".format(FLAGS.checkpoint_file))
                saver.save(sess, FLAGS.checkpoint_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial Learning Rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Minibatch size')
    parser.add_argument('--embedding_file', type=str, default='../data/glove.6B.100d.txt', help='File containing pre trained Glove word embeddings')
    parser.add_argument('--svo_file', type=str, default='../data/open_ie_tuples_data_shuff.txt', help='File containing svo triple + sentence pairs in the form (on each line): subject|verb|object|sentence')
    parser.add_argument('--neg_svo_file', type=str, default='../data/neg_open_ie_tuples_data_shuff.txt', help='File containing negative instance svo triple + sentence pairs in the form (on each line): subject|verb|object|sentence')
    parser.add_argument('--neg_samples', type=int, default=512, help='How many negative samples to use in the sampled softmax objective')
    parser.add_argument('--epochs', type=int, default=1, help='How many passes through the data to make')
    parser.add_argument('--checkpoint_file', type=str, default='../checkpoints/model.ckt')
    parser.add_argument('--max_phrase_size', type=int, default=10, help='The largest size of a phrase in which to average together as input')
    parser.add_argument('--restore_point', type=str, default='tensor_gen_resume/model.ckt')
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--num_queues', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--margin', type=float, default=0.5, help='Margin for the objective') #this is a good margin across models
    parser.add_argument('--reg_lambda', type=int, default=0.00001) #regularization parameter
    parser.add_argument('--predicate_tensor', action="store_true", help='Use the predicate tensor')
    parser.add_argument('--role_factor', action="store_true", help='Use the role factor')
    parser.add_argument('--comp_nn', action="store_true", help='Use the comp neural network')
    ################################################################################################
    # REGULARIZATION PARAMETER VALUES
    # For the Compositional Neural Network, a lambda value of 0.00001 works best
    # For the tensor based models, setting reg_lambda to 0 actually works best (no regularization)
    # Instead use early stopping for regularization for the tensor-based models (early stopping should 
    # also be used for the Compositional Neural Network models as well)
    ##############################################################################################

    FLAGS = parser.parse_args()
    embeddings=Glove(FLAGS.embedding_file)
    instances=EventPredQueuedInstances(FLAGS.svo_file, FLAGS.neg_svo_file, embeddings, FLAGS.num_queues)
    train_prediction_network(instances, embeddings)


