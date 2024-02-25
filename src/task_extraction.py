import spacy
import re

nlp = spacy.load("en_core_web_sm")
text = nlp("Working on a new app development today. Tomorrow, a webinar on UI/UX design. And in a fortnight, a weekend getaway to the mountains.")

connectives = ["then", "tomorrow", "later", "and"]
sentences = [sent.text for sent in text.sents]

def split_into_tasks(sentences, keywords):
    tasks = []
    for sentence in sentences:
        temp_tasks = [sentence]
        for keyword in keywords:
            new_tasks = []
            for task in temp_tasks:
                new_tasks.extend(re.split(r'\b{}\b'.format(re.escape(keyword)), task))
            temp_tasks = new_tasks
        tasks.extend([task.strip() for task in temp_tasks if task.strip() != '' or task.strip() != '"'])
    return tasks

tasks = split_into_tasks(sentences, connectives)
print(tasks)

