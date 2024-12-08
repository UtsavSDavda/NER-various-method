import spacy
from spacy.util import minibatch,compounding
from spacy.training import Example,offsets_to_biluo_tags
import random
from spacy.tokens import DocBin
from tqdm import tqdm
nlp =spacy.load("en_core_web_md")

def is_overlapping(start1, end1, start2, end2):
    return not (end1 <= start2 or end2 <= start1)

def filter_entities(doc, selected_types):
    filtered_entities = []
    for ent in doc.ents:
        if ent.label_ in selected_types:
            filtered_entities.append((ent.start_char, ent.end_char, ent.label_))
    return filtered_entities

def combine_entities(custom_ents, general_ents):
    combined_ents = custom_ents.copy()
    for start, end, label in general_ents:

        if not any(is_overlapping(start, end, ent[0], ent[1]) for ent in custom_ents):
            combined_ents.append((start, end, label))
    return combined_ents

selected_entity_types = ["PERSON", "ORG", "GPE", "DATE", "CARDINAL"]

ner = nlp.get_pipe("ner")
TRAIN_DATA = [ 
('I want an Apple right now.', {'entities': [(10, 15, 'FOOD')]}),
('Can I get a pizza tonight?', {'entities': [(12, 17, 'FOOD')]}),
('I would like to order sushi for dinner today.', {'entities': [(22, 27, 'FOOD')]}),
('Could you please bring me a sandwich?', {'entities': [(28, 36, 'FOOD')]}),
('Do you have any burgers available at 6 PM tomorrow?', {'entities': [(16, 23, 'FOOD')]}),
('I am craving some pasta right now.', {'entities': [(18, 23, 'FOOD')]}),
('Please give me an ice cream.', {'entities': [(18, 27, 'FOOD')]}),
("I'd like to order 32 salads.", {'entities': [(21, 27, 'FOOD')]}),
('Can I have a slice of cake?', {'entities': [(22, 26, 'FOOD')]}),
('Its 9 AM.Is there any pizza left?', {'entities': [(22, 27, 'FOOD')]}),
("I'd love to have a hamburger.", {'entities': [(19, 28, 'FOOD')]}),
('Could you get me some fries?', {'entities': [(22, 27, 'FOOD')]}),
('I want to try the new burger.', {'entities': [(22, 28, 'FOOD')]}),
('Are there any tacos available?', {'entities': [(14, 19, 'FOOD')]}),
('I feel like having a sandwich.', {'entities': [(21, 29, 'FOOD')]}),
('Can you bring me a doughnut?', {'entities': [(19, 27, 'FOOD')]}),
('I would like to have some pizza.', {'entities': [(26, 31, 'FOOD')]}),
('Please get me a bowl of soup.', {'entities': [(24, 28, 'FOOD')]}),
('I would love some pancakes for breakfast.', {'entities': [(18, 26, 'FOOD')]}),
('Can I order a cheeseburger?', {'entities': [(14, 26, 'FOOD')]}),
('I want to try the lasagna.', {'entities': [(18, 25, 'FOOD')]}),
("I'd like a piece of pie.", {'entities': [(20, 23, 'FOOD')]}),
('Can you make me a smoothie?', {'entities': [(18, 26, 'FOOD')]}),
('Is it possible to get a burrito?', {'entities': [(24, 31, 'FOOD')]}),
('Could I have a portion of nachos?', {'entities': [(26, 32, 'FOOD')]}),
("I'd love a bowl of ramen.", {'entities': [(19, 24, 'FOOD')]}),
('Can I get a cup of coffee?', {'entities': [(19, 25, 'FOOD')]}),
('Could you prepare some curry for me?', {'entities': [(23, 28, 'FOOD')]}),
("I'd like to order a steak.", {'entities': [(20, 25, 'FOOD')]}),
('Is there any pizza left?', {'entities': [(13, 18, 'FOOD')]}),
('Can I have 1 slice of pie?', {'entities': [(22, 25, 'FOOD')]}),
('Please serve me a bowl of cereal before 2nd August 2002.', {'entities': [(26, 32, 'FOOD')]}),
('Can you bring me a cup of tea?', {'entities': [(26, 29, 'FOOD')]}),
('I want to order some sushi.', {'entities': [(21, 26, 'FOOD')]}),
('Could I have a donut, please?', {'entities': [(15, 20, 'FOOD')]}),
('Do you have any muffins?', {'entities': [(16, 23, 'FOOD')]}),
("I'd like a plate of pasta.", {'entities': [(20, 25, 'FOOD')]}),
('Can you prepare a salad for me?', {'entities': [(18, 23, 'FOOD')]}),
('Please get me a piece of cake.', {'entities': [(25, 29, 'FOOD')]}),
("I'd love to try your tacos.", {'entities': [(21, 26, 'FOOD')]}),
('Is it possible to get some pizza?', {'entities': [(27, 32, 'FOOD')]}),
('Can I order a bowl of soup from StarBucks?', {'entities': [(22, 26, 'FOOD')]}),
("I'd like a piece of chocolate.", {'entities': [(20, 29, 'FOOD')]}),
('Could you make some guacamole?', {'entities': [(20, 29, 'FOOD')]}),
('I feel like having a cheeseburger.', {'entities': [(21, 33, 'FOOD')]}),
('Can you serve me a bowl of rice?', {'entities': [(27, 31, 'FOOD')]}),
("I'd like a cup of hot chocolate.", {'entities': [(18, 31, 'FOOD')]}),
('Could I order a grill sandwich?', {'entities': [(16, 21, 'SUB_ENTITY'),(22, 30, 'FOOD')]}),
('Is there any apple pie left?', {'entities': [(13, 22, 'FOOD')]})
]

for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

annotated_data = []
for sentence, annotation in TRAIN_DATA:
    custom_entities = annotation["entities"]
    doc = nlp(sentence)
    general_entities = filter_entities(doc, selected_entity_types)
    combined_entities = combine_entities(custom_entities, general_entities)
    annotated_data.append((sentence, {"entities": combined_entities}))

for sentence, annotation in annotated_data:
    print(f"Sentence: {sentence}")
    print(f"Entities: {annotation['entities']}")

examples = [Example.from_dict(nlp.make_doc(text), annotation) for text, annotation in TRAIN_DATA]
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for i in range(20):  
        losses = {}
        random.shuffle(examples)
        for example in examples:
            nlp.update([example], drop=0.5, losses=losses)
        print(f"Epoch {i}, Losses: {losses}")

nlp.to_disk("NERModel")