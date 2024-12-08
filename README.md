# NER-various-method
Perform Named entity recognition using various techniques.
The code currently identifies words having custom labels in a corpususing NLP tehciniques. I have used spacy library for most of the tasks.

**How it works**

1. The EntityRuler File:

   The default Nlp model recognizes approx 18 types of entities. I have called them general entities. If I want to add a few words to be recognized as "LABEL-EXAMPLE", then I can add those particular words as patterns in the EntityRUler model. It is useful when we require EXACT labelling.

2. The EntityRecognizer File:

The EntityRecognizer file has the code that trains the default NLP models to recognize a label "LABEL-EXAMPLE" based on the POSITION of the words attached to that lable. We train the NER model with our own data to make it recognize WHERE can a LABEL be at in a sentence.
For this example, I have used FOOD as a label.
