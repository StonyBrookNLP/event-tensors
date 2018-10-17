# event-tensors
Code and Datasets for the AAAI 2018 paper "Event Representations with Tensor-based compositions"

# Requirements 
* Tensorflow Version 1.4
* Python 3.5

# Running Pretrained Models
The file `get_embeddings.py` gives an example script that loads in a pretrained event-tensor model, and given a dataset of SVO triples (in the same format used for training), runs the SVO triples
through the model to produce event embeddings, and then prints the embeddings to a text file. 

# Preprocessing
We use the Open Information Extraction System [Ollie](https://github.com/knowitall/ollie) to extract triples. The default settings for Ollie will produce triples with long entity and predicate names. 
To shorten these, you need to run Ollie with the OpenParse flag called `expandExtraction` set to false. To do this, you need to create your own main file to run Ollie (replacing [this](https://github.com/knowitall/ollie/blob/master/example/src/main/scala/ollie/Example.scala) example main in the Ollie source) and run it. The main used for parsing the NYT Gigacorpus is provided for reference in the `preproc` directory. Parts of this file will need to be replaced as needed if using a different dataset. 

The above preprocessing step will output a single tuple per line, which wastes quite a bit of space. In order to convert it to the format used in the training the model (one document per line), use the `document_on_line.py` script in the `preproc` directory.
