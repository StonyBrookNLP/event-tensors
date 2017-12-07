########################################################################
#  train_utils
# The file defines various generator classes for processing
# the data. The purpose of these classes is to create an easy iterator
# interface for the data so that little to no processing is needed to 
# feed it to the feed dict
#######################################################################
import numpy as np
import itertools
import math
import string
from collections import deque
import utils.dataset as dataset


FLAGS = None
stopwords = dataset.get_stopwords("./utils/stopwords_full.txt")
translator = str.maketrans('','',string.punctuation)
common_verbs = ['be', 'were', 'been', 'is', "'s", 'have', 'had', 'do', 'did', 'done', 'say', 'said', 'go', 'went', 'gone', 'get', 'got', 'gotton']
auxilary_verbs = ['be', 'am', 'are', 'is', 'was', 'were', 'being', 'been', 'can', 'could', 'dare', 'do', 'does', 'did', 'have', 'has', 'had', 'having', 'may', 'might', 'must', 'ought', 'shall', 'should', 'will', 'would']

# Note, for those new to python, all the classes below are generators. That means to use them, create a class instance:
# a = SingleInstances(foo)
# ...And then iterate over them 
# for i in a: do something with i 
# i should be None at the end of the dataset

class SingleInstances:
    'Class to generate data instances (SVO, Word to predict), generates a word to predict for each word in the sentence the SVO appears in'
    def __init__(self, dataset):
        """
        dataset should be a (chained) ContextualOpenIE_Dataset iterator (defined in datasets.py)
        """
        self.dataset = dataset
        self.processed = 0 #how many instances we have generated

    def __iter__(self):
        for svo in self.dataset:
            if not svo:  #end of the dataset
                yield None
                break
            if svo.verb not in common_verbs and 'said' not in svo.verb:
                for word in svo.sentence.split():
                    word = word.translate(translator)
                    if svo.valid_label(word) and word not in stopwords and len(word) > 2:
                        yield (svo, word)
                        self.processed += 1


#The RandomizedQueue class is used to give some randomness in the batches (in the data, the documents are randomized, but the actual content of the documents 
#are not randomized

#This should be used for the word prediction objective (it will output samples in the format needed for the word prediction task
class RandomizedQueuedInstances:
    'Class to generate only valid data instances, use a queue to add some randomization between the elements in the same document (and thus adding more randomness to the batch' 
    def __init__(self, svo_file, embeddings, num_queues, batch_size, max_phrase_size):
        """
        String svo_file - the text file containing data in the proper format (ie data/ollie_extraction_data_newform_rand_train.txt)
        Int num_queues - the number of queues to use, the more queues, the less performance but the more randomized your batch will be
        Glove embeddings - a Glove embeddings object
        Int batch_size 
        """
        self.embeddings = embeddings
        self.num_queues=num_queues
        self.maxlen=batch_size
        self.queues = [deque(maxlen=self.maxlen) for i in range(num_queues)]
        self.instances=SingleInstances(itertools.chain.from_iterable([iter(dataset.ContextualOpenIE_Dataset(svo_file))] + [itertools.repeat(None, 1)]))
        self.processed = 0 #how many instances we have generated
        self.max_phrase_size=max_phrase_size

    def __iter__(self):
        while True:
            filled = self.fill_queue()
            if filled:  #unload the queues so that the ording is somewhat random (not totally) 
                for q in range(self.num_queues): #which queue to use
                    for _ in range(self.maxlen): 
                        yield self.queues[q].popleft()
            else:
                break
        yield None #end of dataset
    
    def fill_queue(self):
        """
        Fill the set o' queues with some valid data instances
        """
        curr_queue = 0
        num_appends = 0 #keep track of how many times we append, once we reach batchsize*numqueues, we know we have reached the end
        for inst in self.instances:
            if not inst:  #end of the dataset
                return False
                break
            svo = inst[0]
            label = inst[1]
            sub_id, sub_w = self.embeddings.transform(svo.subject.split(), size=self.max_phrase_size)
            verb_id, verb_w = self.embeddings.transform(svo.verb.split(), size=self.max_phrase_size)
            obj_id, obj_w = self.embeddings.transform(svo.obj.split(), size=self.max_phrase_size)
            label_id = self.embeddings.id(label)

            if sub_id is not None and verb_id is not None and obj_id is not None and label_id is not None: 
#                self.queues[curr_queue].append(((sub_id,sub_w), (verb_id, verb_w), (obj_id, obj_w), label_id, svo))
                self.queues[curr_queue].append(((sub_id,sub_w), (verb_id, verb_w), (obj_id, obj_w), label_id))
                curr_queue = (curr_queue + 1) % self.num_queues
                num_appends +=1
                if num_appends == self.maxlen*self.num_queues: #return after we finish filling the queue
                    return True


###Below here are classes used for the predict events objective, since are a bit lengthier...

class Instances: 
    'Class to generate valid OpenIE triples from the dataset, base class for NegativeInstances and InputTargetIterator (positive and negative instances) '
    def __init__(self, dataset, embeddings, max_phrase_size):
        """
        dataset should be a (possibly chained) OpenIE_Dataset iterator, embeddings should be a Glove object
        """
        self.dataset = dataset
        self.processed = 0 #how many instances we have generated
        self.embeddings = embeddings
        self.max_phrase_size=max_phrase_size
    def __iter__(self):
        for svo in self.dataset:
            if not svo:  #end of the dataset
                yield None
                break
            if svo.verb not in common_verbs:
                yield (svo, word)
                self.processed += 1

class NegativeInstances(Instances):
    'Produces negative instances for the predict events objective'
    def __iter__(self):
        for triple in self.dataset:
            if not triple:
                yield None
                break

            sub_id, sub_w = self.embeddings.transform(triple.subject.split(), size=self.max_phrase_size)
            verb_id, verb_w = self.embeddings.transform(triple.verb.split(), size=self.max_phrase_size)
            obj_id, obj_w = self.embeddings.transform(triple.obj.split(), size=self.max_phrase_size)

            if triple.verb not in common_verbs and 'said' not in triple.verb and sub_id is not None and verb_id is not None and obj_id is not None:  #check if we have a word embedding for this, and if its not common verb or said varient
                instance = (triple, ((sub_id, sub_w), (verb_id, verb_w), (obj_id, obj_w)))
                yield instance
                self.processed +=1

class InputTargetIterator:
    'Class to generate pairs of tuples in the same document within a certain window (this produces the input and target instance for predict event objective'
    def __init__(self,dataset, embeddings, max_phrase_size, window=1):
        """DocOpenIE_Dataset dataset 
           Glove embeddings
        """
        self.documents=dataset
        self.processed=0
        self.embeddings=embeddings
        self.max_phrase_size=max_phrase_size
        self.window=window

    def get_ids(self, triple):
        sub_id, sub_w = self.embeddings.transform(triple.sub.split(), size=self.max_phrase_size)
        verb_id, verb_w = self.embeddings.transform(triple.verb.split(), size=self.max_phrase_size)
        obj_id, obj_w = self.embeddings.transform(triple.obj.split(), size=self.max_phrase_size)
        instance = ((sub_id, sub_w), (verb_id, verb_w), (obj_id, obj_w))

        if sub_id is not None and verb_id is not None and obj_id is not None:  #check if we have a word embedding for this
            return instance
        else:
            return None

    def __iter__(self):
        for doc in self.documents:
            if not doc:
                yield None
                break
            tuples = []
            for sent in doc.sentences:
                tuples += sent.tuples
            tuples_data = [(x, self.get_ids(x)) for x in tuples if self.get_ids(x) is not None]
            if len(tuples_data) > self.window:
                for i in range(len(tuples_data) - self.window):
                    input_tuple = tuples_data[i][0] 
                    input_tuple_value = tuples_data[i][1]
                    if input_tuple.verb not in common_verbs and 'said' not in input_tuple.verb: #filter out stop events
                        for j in range(1, self.window + 1): #get the targets (ie the next tuples)
                            target = tuples_data[i+j][0]
                            target_value = tuples_data[i+j][1]
                            if target.verb not in common_verbs and 'said' not in target.verb:  #filter out stop events
                                instances = (input_tuple_value, target_value, input_tuple, target) 
                                yield instances
                                self.processed += 1

class EventPredQueuedInstances(RandomizedQueuedInstances):
    'This class is like RandomizedQueuedInstances, with a couple changes so it can be used for event prediction objective'
    def __init__(self, svo_file, neg_svo_file, embeddings, num_queues, batch_size):
        """
        dset should be a (possibly chained) Dataset iterator
        embeddings should be a Glove object
        """
        self.embeddings = embeddings
        self.num_queues=num_queues
        self.maxlen=batch_size
        self.queues = [deque(maxlen=self.maxlen) for i in range(num_queues)]
        self.instances=iter(InputTargetIterator(dataset.DocOpenIE_Dataset(svo_file), embeddings))
        self.neg_instances=iter(NegativeInstances(dataset.OpenIE_Dataset(neg_svo_file), embeddings))
        self.processed = 0 #how many instances we have generated
    
    def fill_queue(self):
        """
        Fill me set o' queues with some delicious and valid data instances (now high is Riboflavin!)
        """
        curr_queue = 0
        num_appends = 0 #keep track of how many times we append, once we reach batchsize*numqueues, we know we have reached the end
        for inst in self.instances:
            if not inst:  #end of the dataset
                return False
                break
            input_tuple_value = inst[0]
            target_tuple_value = inst[1]
            neg_inst = next(self.neg_instances)[1]

            self.queues[curr_queue].append((input_tuple_value, target_tuple_value, neg_inst))
            curr_queue = (curr_queue + 1) % self.num_queues
            num_appends +=1
            if num_appends == self.maxlen*self.num_queues: #return after we finish filling the queue
                return True




