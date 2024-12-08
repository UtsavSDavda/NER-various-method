#This is the file where we add patterns of labels to the EntityRuler model.
import spacy
from spacy.matcher import Matcher
import pandas as pd
import numpy as np
from spacy.pipeline import EntityRuler
def is_string(value):
    if value == '-1':
        return np.nan 
    else:
        return value if isinstance(value, str) else np.nan
nlp = spacy.load('en_core_web_sm')
data = pd.read_csv('indian_food.csv')
df1 = pd.DataFrame(data)
categories = list(data.columns)
categories.remove('cook_time')
categories.remove('prep_time')
print(categories)
for ct in ['name','ingredients']:
    df1[ct] = df1[ct].apply(is_string)
    df1[ct].dropna(inplace=True,axis=0)
df1.dropna(inplace=True,axis=0)

multi_patterns = [
    [],
    [],
    [],
    []
]

multi_list = []
array2 = []
for cat in categories:
    unique_values = list(df1[cat].unique())
    print(unique_values)
    for i in range(len(unique_values)):
        array1 = [word.lower() for word in (str(unique_values[i]).split(' '))]
        multi_list.append(array1)
print(multi_list[0:10])
multi_list_patterns = []
for m in multi_list:
    multi_list_patterns.append({"label":"FOOD","pattern":[{"LOWER":word.lower()} for word in m]})
ruler = nlp.add_pipe("entity_ruler",before="ner")
ruler.add_patterns(multi_list_patterns)
matcher = Matcher(nlp.vocab)
stop_flag = False
nlp.to_disk("RulerModel")
while(stop_flag == False):
    input_text = input("Enter your text")
    doc = nlp(input_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(entities)
    stop = input("Press Q to quit")
    if stop == "q":
        stop_flag = True
