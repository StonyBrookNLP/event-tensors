#################################################
# Helper functions for loading pretrained glove
# embeddings
################################################
import numpy as np

class Glove:
    'class for loading and handling pretrained glove embeddings'
    
    def __init__(self, filename):
        self.filename = filename
        self.vocab_id, self.embd = self.load(self.filename)
        self.vocab_size = len(self.vocab_id)
        self.dim = self.embd.shape[1]

    def load(self, filename):
        with open(filename, 'r') as fi:
            embd = []
            vocab_id = {}
            for line in fi:
                data = line.strip().split()
                vocab_id[data[0].strip()] = len(vocab_id)
                str_em = data[1:]
                embd.append([float(x) for x in str_em])
        embd = np.array(embd)
        assert len(vocab_id) == embd.shape[0]
        return vocab_id, embd

    def reverse_dict(self):
        """
        Return a id to vocab dict
        """
        return dict(zip(self.vocab_id.values(), self.vocab_id.keys()))

    def id(self, word):
        """return the vocab id for word"""
        if word in self.vocab_id:
            return self.vocab_id[word]
        else:
            return None

    def embedding(self, word):
        """return the embedding for the word"""
        assert word in self.vocab_id
        return self.embd[self.id(word), :]

    def average_hot(self, words): 
        """Get a n hot vector for a list of n words, where entries corresponding to the ids
            of the words in the list passed in all have 1/n (so when you multiply them by embeddings matrix
            it gives you the average)
        """
        words = [x for x in words if x in self.vocab_id] #only take those in our vocabulary
        if words:
            n = len(words)
            hot = np.zeros(self.vocab_size)
            for i in range(n):
                index = self.id(words[i])
                hot[index] = 1/n
            return hot
        else:
            return None
                

    def average_embed(self, words):
        """return the average embedding for the list of words indicated"""
        words = [x for x in words if x in self.vocab_id] #only take those in our vocabulary
        if words:
            toavg = np.array([self.embedding(x) for x in words])
            avg = np.mean(toavg, axis=0)
            return avg
        else:
            return None

    def transform(self, words, size=None):
        """Transform list of words to list of vocab ids
           Also return the weights for the embeddings
        """
        if not size:
            return np.array([self.id(x) for x in words if x in self.vocab_id])
        else:
            if len(words) > size:
                words = words[:size]
            words = [self.id(x) for x in words if x in self.vocab_id]
            if words:
                ar = np.array(words)
                padded = np.pad(ar,pad_width=(0,size-len(words)), mode='constant', constant_values=-1)
                weights = (padded+1 != 0).astype(int)
                weights = weights / len(words)
                padded = np.absolute(padded)
                return padded, weights
            else:
                return None, None
        #words = [self.id(x) for x in words if x in self.vocab_id] #word ids
        #if words:
        #    ar = np.array(words)
        #    weights = np.zeros(ar.shape[0]) + (1/ar.shape[0])
        #    return ar, weights
        #else:
        #    return None, None

    def transform2(self, words):
        """Transform list of words to list of vocab ids
           Also return the weights for the embeddings
        """
        words = [self.id(x) for x in words if x in self.vocab_id] #word ids
        if words:
            ar = np.array(words)
            weights = np.zeros(ar.shape[0]) + (1/ar.shape[0])
            return ar, weights
        else:
            return None, None

        


