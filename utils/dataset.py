############################################
#   Dataset object, for efficiently streaming the
#   SVO triple data into a TensorFlow
#   feed
############################################
import os
import itertools
from collections import namedtuple
common_verbs = ['be', 'were', 'been', 'is', "'s", 'have', 'had', 'do', 'did', 'done', 'say', 'said', 'go', 'went', 'gone', 'get', 'got', 'gotton']
auxilary_verbs = ['be', 'am', 'are', 'is', 'was', 'were', 'being', 'been', 'can', 'could', 'dare', 'do', 'does', 'did', 'have', 'has', 'had', 'having', 'may', 'might', 'must', 'ought', 'shall', 'should', 'will', 'would', "'s"]
prepositions = ['of', 'to', 'in', 'for', 'on', 'by', 'at', 'with', 'from', 'into', 'during', 'including', 'until', 'about', 'like', 'through', 'over', 'before', 'after', 'around', 'near', 'above', 'without', 'but', 'up', 'down']
SENT_SEP="|SENT|"
TUP_SEP="|TUP|"

SimpleTuple = namedtuple('SimpleTuple', 'sub verb obj')

class Dataset:  
    'regular svo triple input format'
    def __init__(self, filename, calc_samples=False):
        self.filename=filename
        if calc_samples:
            self.size = self.count(filename)
        else:
            self.size = -1

    def __iter__(self):  #generates a svo triple along with a sentence it appears in
        with open(self.filename, 'r') as fi:
            for line in fi:
                splits = line.split('|')
                if len(splits) == 4:
                    yield(SVO(splits[0], splits[1], splits[2], splits[3]))
        return None #return None for the last item

    def count(self, filename):
        x = 0
        with open(self.filename, 'r') as fi:
            for line in fi:
                x += 1
        return x

class OpenIE_Dataset:  
    'Corpus with NYT Giga corpus directory structure, class that allows iteration over all files in the corpus (for Old Form of OpenIE dataset, similar to Dataset, but for openie)'

    def __init__(self, filename):
        self.filename=filename

    def __iter__(self):  #generates a svo triple along with a sentence it appears in
        with open(self.filename, 'r', encoding='utf-8') as fi:
            for line in fi:
                splits = line.split('|')
                if len(splits) == 6:
                    verb = splits[3].strip().lower()
                    vlist = [x for x in verb.split() if x not in auxilary_verbs]
                    if vlist:
                        v = " ".join(vlist)
                        yield(OpenIE_Triple(splits[0], splits[1], splits[2], v, splits[4], splits[5]))
        return None #return None for the last item

class Sentence:
    def __init__(self, sentid, text, tuples):
        """
        sentid is an integer id, text is the sentence text itself, and tuples
        is a list of SimpleTuples containing the tuples found in the sentence
        """
        self.sent_id=sentid
        self.text=text
        self.tuples=tuples

    def __str__(self):
        string = "%s:%s\n" % (self.sent_id, self.text)
        for i in self.tuples:
            string+="\t(%s, %s, %s)\n" % (i.sub, i.verb, i.obj)
        return string
 
class Document:
    def __init__(self, doc_id, sentences):
        """
        sentences is a list of Sentence objects
        """
        self.doc_id=doc_id
        self.sentences=sentences

    def __str__(self):
        string = "***********%s************" % self.doc_id
        for i in self.sentences:
            string += str(i)
        return string


class DocOpenIE_Dataset:
    'For the document level version of the openie dataset, where each line stores a document, yield Document objects' 
    #Also the iterator in this class removes some of the redundent expressions ollie extracts
    def __init__(self,filename):
        self.filename=filename

    def __iter__(self):
        with open(self.filename, 'r', encoding='utf-8') as fi:
            for doc in fi: #each line is a document
                doc_sents = doc.split(SENT_SEP)
                doc_id = ""
                sents_list = []
                for sents in doc_sents:
                    sent_tups = sents.split(TUP_SEP)
                    if len(sent_tups) >= 2:
                        info = sent_tups[0].split("|")
                        tuples = sent_tups[1:]
                        if len(info) == 3: 
                            doc_id=info[0]
                            sent_id=info[1]
                            sent_text=info[2]
                            tuples_list = []
                            prev_verbs = []
                            for tups in tuples:
                                splits=tups.split("|")
                                if len(splits) < 3:
                                    continue
                                verb = splits[1].strip().lower()
                                vlist = [x for x in verb.split() if x not in auxilary_verbs] #remove the auxilariry verbs (so remember, during test time we should also remove verbs)
                                if vlist:
                                    v = " ".join(vlist)
                                    main_verb = " ".join([x for x in vlist if x not in prepositions])
                                    if not any([main_verb in x for x in prev_verbs]): #make sure this tuple doesnt have the same main verb as any previous ones 
                                        tuples_list.append(SimpleTuple(sub=splits[0].strip().lower(), verb=v, obj=splits[2].strip().lower()))  #add tuple to this sentence, keep prepositions with verb
                                        prev_verbs.append(main_verb)   #since this verb was used, add it to list of previously used main verbs

                                    #tuples_list.append(SimpleTuple(sub=splits[0], verb=v, obj=splits[2])) #uncomment this and comment the above 4 lines if using regular not openie
                            if tuples_list:
                                sents_list.append(Sentence(sent_id, sent_text, tuples_list))
                        else:
                            print("Info parsing error")
                    else:
                        print("sent tups parsing error")
                if sents_list:
                    yield Document(doc_id, sents_list)
                    

class ContextualOpenIE_Dataset: 
    """
    Yield ContextualOpenIE tuples (there just OpenIE tuples with the sentence before and after stored with them, the name
    is just to make them sound fancy)
    """
    def __init__(self,filename):
        self.filename=filename
        self.dataset = DocOpenIE_Dataset(filename)

    def __iter__(self):
        for doc in self.dataset:
            docid = doc.doc_id
            sents = doc.sentences
            num_sents = len(sents)
            for i, s in enumerate(sents):
                sentid = s.sent_id
                if i != 0 and i != num_sents-1:
                    prev_sent = sents[i-1].text
                    next_sent = sents[i+1].text
                    curr_sent = s.text
                elif i != 0 and i == num_sents-1: #at the end
                    prev_sent = sents[i-1].text
                    next_sent = ""
                    curr_sent = s.text
                elif i == 0 and i != num_sents-1: #at beginning
                    prev_sent = ""
                    next_sent = sents[i+1].text
                    curr_sent = s.text
                else:
                    prev_sent = ""
                    next_sent = ""
                    curr_sent = s.text
                for x in s.tuples:
                    yield ContextOpenIE_Triple(docid, sentid, x.sub, x.verb, x.obj, curr_sent, prev_sent, next_sent)

def chain_dataset(filename, num):
    "Return series of Dataset iterators chained together via itertools, chain together as many epochs needed - TESTED and works"
    return itertools.chain.from_iterable([iter(Dataset(filename)) for x in range(num)] + [itertools.repeat(None, 1)])

def chain_dataset2(filename, num):
    "Return series of OpenIE_Dataset iterators chained together via itertools, chain together as many epochs needed - TESTED and works"
    return itertools.chain.from_iterable([iter(OpenIE_Dataset(filename)) for x in range(num)] + [itertools.repeat(None, 1)])



class SVO:
    'A Subject verb object triple along with the sentence it appears in'
    def __init__(self, subject, verb, obj, sentence):
        self.subject =subject.strip().lower()
        self.verb = verb.strip().lower()
        self.obj = obj.strip().lower()
        self.sentence = sentence.strip().lower()

    def __str__(self):
        return "(%s, %s, %s) - %s" % (self.subject, self.verb, self.obj, self.sentence)

    def valid_label(self, label):
        """
        Check if a word is a valid label to predict (ie
        it is not part of the SVO text)
        """
        sub = self.subject.split()
        obj = self.obj.split()
        verb = self.verb.split()
        return not (label in sub or label in obj or label in verb)

class OpenIE_Triple(SVO):
    'A Subject verb object triple along with the sentence it appears in'
    def __init__(self, doc_id, sent_id, subject, verb, obj, sentence):
        super().__init__(subject, verb, obj, sentence)
        self.doc_id = doc_id.strip()
        self.sent_id = sent_id.strip()

    def __str__(self):
        return "(%s, %s, %s, %s, %s) - %s" % (self.doc_id, self.sent_id, self.subject, self.verb, self.obj, self.sentence)

class ContextOpenIE_Triple(OpenIE_Triple):
    def __init__(self, doc_id, sent_id, subject, verb, obj, sentence, before_sent, after_sent):
        super().__init__(doc_id, sent_id, subject, verb, obj, sentence)
        self.before_sent = before_sent.strip().lower()
        self.after_sent = after_sent.strip().lower()

    def __str__(self):
        return "(%s, %s, %s, %s, %s) - %s \n %s \n %s \n" % (self.doc_id, self.sent_id, self.subject, self.verb, self.obj, self.sentence, self.before_sent, self.after_sent)

def get_stopwords(filename="stopwords.txt"):
    stopwords = []
    with open(filename, 'r') as fi:
        for line in fi:
            stopwords.append(line.strip())
    return stopwords



