######################################
# Code to train the word prediction 
# objective (contains code for the compositional nn,
# role factor, and predicate tensor)
######################################
import tensorflow as tf
import numpy as np
import utils.dataset as dataset
from utils.glove_utils import Glove
from utils.train_utils import RandomizedQueuedInstances
import argparse
import itertools
import math
import string
from collections import deque

FLAGS = None

def placeholder_inputs_ft(embeddings):
    """
    Glove embeddings
    """
    embed_dim = embeddings.vocab_size
    subject_ph = tf.placeholder(tf.int64, shape=(FLAGS.batch_size * FLAGS.max_phrase_size), name="a")
    verb_ph = tf.placeholder(tf.int64, shape=[FLAGS.batch_size * FLAGS.max_phrase_size], name="b")
    object_ph = tf.placeholder(tf.int64, shape=[FLAGS.batch_size * FLAGS.max_phrase_size], name="c")
    label_ph = tf.placeholder(tf.int64, shape=(FLAGS.batch_size, 1), name="label")

    #Since we are using tf.embedding_lookup_sparse we need the to pass in extra sparse weights as well
    #This might make more sense if you look up how embedding_lookup_sparse words 
    subject_w = tf.placeholder(tf.float32, shape=(FLAGS.batch_size * FLAGS.max_phrase_size), name="d")
    verb_w = tf.placeholder(tf.float32, shape=(FLAGS.batch_size * FLAGS.max_phrase_size), name="e")
    object_w = tf.placeholder(tf.float32, shape=(FLAGS.batch_size * FLAGS.max_phrase_size), name="f")

    return subject_ph, verb_ph, object_ph, label_ph, subject_w, verb_w, object_w

def fill_feed_dict_ft(instance_iter, subject_ph, verb_ph, object_ph, label_ph, subject_w_ph, verb_w_ph, object_w_ph, embeddings):

    """
    Obtain the next batch from the instance iter and fill the feed dict to use with the model
    RandomizedQueuedInstances instance_iter - The instance iterator
    tf.placeholder subject_ph - A place holder object (returned from placeholder_inputs_ft)
    Glove embeddings

    Return the feed dict as well as a flag indicating if we are finished
    """

    shape = np.array([FLAGS.batch_size, embeddings.vocab_size])
    embed_dim = embeddings.dim
    batch_label = np.zeros((FLAGS.batch_size, 1))

    #indices for sparse array
    sub_indices = []
    verb_indices = []
    obj_indices = []

    sub_id_values = []
    sub_weight_values = []
    verb_id_values = []
    verb_weight_values = []
    obj_id_values = []
    obj_weight_values = []


    done = False
    for i in range(FLAGS.batch_size):
        inst = next(instance_iter)
        if inst:
            sub_id, sub_w = inst[0]
            verb_id, verb_w = inst[1]
            obj_id, obj_w = inst[2]
            label_id = inst[3]

            batch_label[i, :] = label_id 

            sub_id_values.extend(sub_id)
            sub_weight_values.extend(sub_w)
            verb_id_values.extend(verb_id)
            verb_weight_values.extend(verb_w)
            obj_id_values.extend(obj_id)
            obj_weight_values.extend(obj_w)

        else: #reached the end of instances
            done = True
            break

    sub_indices = np.array(sub_indices)
    verb_indices = np.array(verb_indices)
    obj_indices = np.array(obj_indices)
    sub_weight_values = np.array(sub_weight_values)
    verb_weight_values = np.array(verb_weight_values)
    obj_weight_values = np.array(obj_weight_values)
    sub_id_values = np.array(sub_id_values)
    verb_id_values = np.array(verb_id_values)
    obj_id_values = np.array(obj_id_values)


    feed_dict = {
      subject_ph: sub_id_values,
      verb_ph: verb_id_values,
      object_ph: obj_id_values,
      label_ph: batch_label,
      subject_w_ph: sub_weight_values,
      verb_w_ph: verb_weight_values,
      object_w_ph: obj_weight_values
    }
    return feed_dict,done


def additive_nn(subject_ph, verb_ph, object_ph, label_ph, subject_ph_w, verb_ph_w, object_ph_w, embeddings, indices):
    """
    Compositional Neural Network Baseline model, this function creates the graph needed 
    Create the representation of the entire event using a straightforward 2 layer nn model
    Pass in the placeholders, the embeddings object, and indices which is needed for sparse embeddings lookup, indices should be created via
    indices = np.array([[b,x] for b in range(FLAGS.batch_size) for x in range(FLAGS.max_phrase_size)], dtype=np.int64)
    """ 
    #The main network, compute the svo representation
    embed_dim = embeddings.dim 
    embd = embeddings.embd.astype(np.float32)
    embed_layer = tf.Variable(embd, expected_shape=[embeddings.vocab_size, embed_dim], name="Embed")
    W = tf.Variable(tf.truncated_normal([3*embed_dim,FLAGS.hidden_size], stddev=1.0 / math.sqrt(embed_dim)), name='W') #the encoding matrix, take in concatenation of svo
    H = tf.Variable(tf.truncated_normal([FLAGS.hidden_size,embed_dim], stddev=1.0 / math.sqrt(embed_dim)), name='H') #output matrix, take in hidden layer and return representation

    print("Build network")
    #biases reduce the performance (on all models actually) so I have left them off all models, they are pretty easy to put back in 

    #hidden_biases = tf.Variable(tf.zeros([1, FLAGS.hidden_size]), name='hidden_Biases') 

    nce_weights = tf.Variable(tf.truncated_normal([embeddings.vocab_size, embed_dim], stddev=1.0 / math.sqrt(embed_dim)), name='NCE_W')
    nce_biases = tf.Variable(tf.zeros([embeddings.vocab_size]), name='NCE_Biases')

    subject_indices = tf.SparseTensor(indices, subject_ph, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    verb_indices = tf.SparseTensor(indices, verb_ph, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    object_indices = tf.SparseTensor(indices, object_ph, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    subject_weights = tf.SparseTensor(indices, subject_ph_w, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    verb_weights = tf.SparseTensor(indices, verb_ph_w, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))
    object_weights = tf.SparseTensor(indices, object_ph_w, np.array((FLAGS.batch_size, FLAGS.max_phrase_size), dtype=np.int64))

    #Average together multi word phrases
    subject_avg = tf.nn.embedding_lookup_sparse(embed_layer, subject_indices, subject_weights, combiner='sum') 
    verb_avg = tf.nn.embedding_lookup_sparse(embed_layer, verb_indices, verb_weights, combiner='sum')
    object_avg = tf.nn.embedding_lookup_sparse(embed_layer, object_indices, object_weights, combiner='sum')

    verb_foo = tf.reshape(verb_avg, shape=[FLAGS.batch_size, embed_dim])
    subject_foo = tf.reshape(subject_avg, shape=[FLAGS.batch_size, embed_dim])
    object_foo = tf.reshape(object_avg, shape=[FLAGS.batch_size, embed_dim])

    svo = tf.concat([verb_foo, subject_foo, object_foo], 1) #need to make it batchsizeXdim for multiplication
    
   # hidden = tf.tanh(tf.matmul(svo, W) + hidden_biases) #hidden layer with biases

    hidden = tf.tanh(tf.matmul(svo, W)) #hidden layer 
    final = tf.matmul(hidden, H, name='final')


    #Try to predict the context, use sampled softmax (real softmax is much too slow, sampled softmax works quite well here

    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights,biases=nce_biases,labels=label_ph,inputs=final,num_sampled=FLAGS.neg_samples,num_classes=embeddings.vocab_size), name="loss") + FLAGS.reg_lambda*tf.nn.l2_loss(W) + flags.reg_lambda*tf.nn.l2_loss(H) 


    return loss

def role_factor_network(subject_ph, verb_ph, object_ph, label_ph, subject_ph_w, verb_ph_w, object_ph_w, embeddings, indices):
    """
    The Role Factored Tensor Model
    """ 
    #The main network, compute the svo representation
    embed_dim = embeddings.dim 
    embd = embeddings.embd.astype(np.float32)
    embed_layer = tf.Variable(embd, expected_shape=[embeddings.vocab_size, embed_dim], name="Embed") 
    #You need to initialize with std=1/embed_dim, not square root, according to Xavier initialization
    tensor = tf.Variable(tf.truncated_normal([embed_dim, embed_dim, embed_dim], stddev=1.0 / embed_dim), name='tensor')

    W = tf.Variable(tf.truncated_normal([2*embed_dim,embed_dim], stddev=1.0 / math.sqrt(embed_dim)), name='W') #the encoding matrix, take in concatenation of svo

    nce_weights = tf.Variable(tf.truncated_normal([embeddings.vocab_size, embed_dim], stddev=1.0 / math.sqrt(embed_dim)), name='NCE_W')
    nce_biases = tf.Variable(tf.zeros([embeddings.vocab_size]), name='NCE_Biases')

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

    vs = tf.einsum('ijk,ck,cj->ci', tensor, subject_foo, verb_foo) #get the immediatary representations
    vo = tf.einsum('ijk,ck,cj->ci', tensor, object_foo, verb_foo)

    svo = tf.concat([vs,vo], 1) 
    print("dim")
    print(svo.get_shape())
    final = tf.matmul(svo, W, name='final')

    #Try to predict the context, use negative sampling loss

    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights,biases=nce_biases,labels=label_ph,inputs=final,num_sampled=FLAGS.neg_samples,num_classes=embeddings.vocab_size), name="loss")
    #I have just taken off lambda here, it can be added back again if needed

    return loss

def predicate_tensor_network(subject_ph, verb_ph, object_ph, label_ph, subject_ph_w, verb_ph_w, object_ph_w, embeddings, indices):
    """
    Define the graph for the predicate_tensor_network
    """ 
    #The main network, compute the svo representation
    embed_dim = embeddings.dim 
    embd = embeddings.embd.astype(np.float32)
    embed_layer = tf.Variable(embd, expected_shape=[embeddings.vocab_size, embed_dim], name="Embed")
    #Again, make sure tensors are init with stddev 1/ embed_dim
    W = tf.Variable(tf.truncated_normal([embed_dim, embed_dim, embed_dim], stddev=1.0 / embed_dim), name='W')
    U = tf.Variable(tf.truncated_normal([embed_dim, embed_dim, embed_dim], stddev=1.0 / embed_dim), name='U')

    nce_weights = tf.Variable(tf.truncated_normal([embeddings.vocab_size, embed_dim], stddev=1.0 / math.sqrt(embed_dim)), name='NCE_W')
    nce_biases = tf.Variable(tf.zeros([embeddings.vocab_size]), name='NCE_Biases')

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

    #This looks REALLY wierd, I know, but this is actually equivalent to the formula for the predicate tensor,
    #However it can be computed a little quicker by tensorflow (at the cost of needed some stack and unstack operations unfortunatly)
    obs = tf.stack([tf.diag(x) for x in tf.unstack(object_foo, axis=0)], axis=2)
    alpha = tf.einsum('ijk,ci->jkc', W, verb_foo)
    gamma = tf.einsum('iak,ajk->ijk', obs, alpha)
    L = tf.einsum('ijk,jkc->ikc', U, gamma)
    final = tf.einsum('jic,cj->ci', L, subject_foo) #ROWS are the embeddings for the batch

    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights,biases=nce_biases,labels=label_ph,inputs=final,num_sampled=FLAGS.neg_samples,num_classes=embeddings.vocab_size), name="loss")
#    tf.summary.scalar('loss', loss)

    return loss


def train_network_with_embeddings(instances, embeddings):
    """
    Train the network and finetune the word embeddings
    """
    print("Training with embeddings")
    inst_iter=iter(instances)
    indices = np.array([[b,x] for b in range(FLAGS.batch_size) for x in range(FLAGS.max_phrase_size)], dtype=np.int64)
    sub_ph, verb_ph, obj_ph, label_ph, sub_ph_w, verb_ph_w, obj_ph_w = placeholder_inputs_ft(embeddings)
    if FLAGS.role_factor:
        loss = role_factor_network(sub_ph, verb_ph, obj_ph, label_ph, sub_ph_w, verb_ph_w, obj_ph_w, embeddings, indices)
    elif FLAGS.predicate_tensor:
        loss = predicate_tensor_network(sub_ph, verb_ph, obj_ph, label_ph, sub_ph_w, verb_ph_w, obj_ph_w, embeddings, indices)
    else:
        loss = additive_nn(sub_ph, verb_ph, obj_ph, label_ph, sub_ph_w, verb_ph_w, obj_ph_w, embeddings, indices)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss, name="optimize", global_step=global_step)
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
            
            feed_dict, done = fill_feed_dict_ft(inst_iter, sub_ph, verb_ph, obj_ph, label_ph, sub_ph_w, verb_ph_w, obj_ph_w, embeddings)
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
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial Learning Rate') #This learning rate works good across all methods
    parser.add_argument('--batch_size', type=int, default=128, help='Minibatch size')
    parser.add_argument('--embedding_file', type=str, default='data/glove.6B.100d.txt', help='File containing pre trained Glove word embeddings')
    parser.add_argument('--svo_file', type=str, default='data/ollie_extraction_data_newform_rand_dev.txt', help='Main training data') #dont use this default if actaully training, this is just an example
    parser.add_argument('--neg_samples', type=int, default=512, help='How many samples to use in the sampled softmax objective')
    parser.add_argument('--epochs', type=int, default=1, help='How many passes through the data to make')
    parser.add_argument('--checkpoint_file', type=str, default='../checkpoints/model.ckt', help='Where to save the model')
    parser.add_argument('--max_phrase_size', type=int, default=10, help='The largest size of a phrase in which to average together as input')
    parser.add_argument('--restore_point', type=str, default='Where to restore the model from, if resume is true')
    parser.add_argument('--resume', action="store_true", help='Set to true if restoring the model')
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--num_queues', type=int, default=256, help='number of queues to use in the RandomizedQueuedInstance')
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
    instances=RandomizedQueuedInstances(FLAGS.svo_file, embeddings, FLAGS.num_queues, FLAGS.batch_size, FLAGS.max_phrase_size)
    train_network_with_embeddings(instances, embeddings)


