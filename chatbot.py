import pandas as pd
import os, shutil, datetime
from difflib import SequenceMatcher

class Chatbot:

    def __init__(self, nomatch, qafile, thankyou):
        self.nomatch = nomatch
        self.qafile =   qafile
        self.thankyou = thankyou
        self.logfile = 'chat.log'

    def run_bot(self):
        qadata = pd.read_csv(self.qafile, skipinitialspace = True, quotechar = '"')
        self.questions = qadata['Question']
        self.answers = qadata['Answer']


    def levenstein_distance(self, sentence1, sentence2):
        distance = jellyfish.levenshtein_distance(sentence1, sentence2)
        normalized_distance = distance / max(len(sentence1), len(sentence2))
        return 1.0 - normalized_distance

    def sequence_matcher_distance(self, sentence1, sentence2):
        return SequenceMatcher(None, sentence1, sentence2).ratio()

    def get_highest_similarity(self, customer_question):
        similarity_threshold = 0.3
        max_similarity = 0
        highest_prob_index = 0
        for question_id in range(len(self.questions)):
            similarity = self.sequence_matcher_distance(customer_question, self.questions[question_id])
            if similarity > max_similarity:
                highest_index = question_id
                max_similarity = similarity
        if max_similarity > similarity_threshold:
            return self.answers[highest_index]
        else:
            return self.nomatch

    def get_response(self, question):
        if (question == "bye"):
            answer = self.thankyou
        else:
            answer = self.get_highest_similarity(question)
        line = '[{}] {}: {}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), question, answer)
        with open(self.logfile, "a+") as m:
            m.write(line)
        return answer

