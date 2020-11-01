import pandas as pd
import os, shutil, datetime
from difflib import SequenceMatcher
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem.snowball import SnowballStemmer
import random
from nltk.classify import SklearnClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

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
    
    def extract_tagged(self, sentences):
        features = []
        for tagged_word in sentences:
            word, tag = tagged_word
            if tag=='NN' or tag == 'VBN' or tag == 'NNS' or tag == 'VBP' or tag == 'RB' or tag == 'VBZ' or tag == 'VBG' or tag =='PRP' or tag == 'JJ':
                features.append(word)
        return features
    
    def extract_feature(self, text):
        lmtzr = WordNetLemmatizer()
        words = preprocess(text)
    #     print('words: ',words)
        tags = nltk.pos_tag(words)
    #     print('tags: ',tags)
        extracted_features = extract_tagged(tags)
    #     print('Extracted features: ',extracted_features)
        stemmed_words = [stemmer.stem(x) for x in extracted_features]
    #     print(stemmed_words)

        result = [lmtzr.lemmatize(x) for x in stemmed_words]
        return result
    
    def word_feats(self, words):
        return dict([(word, True) for word in words])
    
    def extract_feature_from_doc(self, data):
        result = []
        corpus = []
        # The responses of the chat bot
        answers = {}
        for (text,category,answer) in data:

            features = extract_feature(text)

            corpus.append(features)
            result.append((word_feats(features), category))
            answers[category] = answer

        return (result, sum(corpus,[]), answers)
    
    def extract_feature_from_doc(self, data):
        result = []
        corpus = []
        # The responses of the chat bot
        answers = {}
        for (text,category,answer) in data:

            features = extract_feature(text)

            corpus.append(features)
            result.append((word_feats(features), category))
            answers[category] = answer

    return (result, sum(corpus,[]), answers)

    def run_ml_based_bot(self, input):
        data = get_content(qafile)
        features_data, corpus, answers = extract_feature_from_doc(data)
        training_data, test_data = split_dataset(features_data, split_ratio)
        naive_bayes_response(input, training_data, test_data)
        
   def split_dataset(self, data, split_ratio):
        random.shuffle(data)
        data_length = len(data)
        train_split = int(data_length * split_ratio)
        return (data[:train_split]), (data[train_split:])
    
    def train_using_decision_tree(self, training_data, test_data):    
        classifier = nltk.classify.DecisionTreeClassifier.train(training_data, entropy_cutoff=0.6, support_cutoff=6)
        classifier_name = type(classifier).__name__
        training_set_accuracy = nltk.classify.accuracy(classifier, training_data)
        print('training set accuracy: ', training_set_accuracy)
        test_set_accuracy = nltk.classify.accuracy(classifier, test_data)
        print('test set accuracy: ', test_set_accuracy)
        return classifier, classifier_name, test_set_accuracy, training_set_accuracy
    
    def train_using_naive_bayes(self, training_data, test_data):
        classifier = nltk.NaiveBayesClassifier.train(training_data)
        classifier_name = type(classifier).__name__
        training_set_accuracy = nltk.classify.accuracy(classifier, training_data)
        test_set_accuracy = nltk.classify.accuracy(classifier, test_data)
        return classifier, classifier_name, test_set_accuracy, training_set_accuracy
    
    def naive_bayes_response(self, training_data, test_data, input_sentence):
        classifier, classifier_name, test_set_accuracy, training_set_accuracy = train_using_naive_bayes(training_data, test_data)
        print(training_set_accuracy)
        print(test_set_accuracy)
        print(len(classifier.most_informative_features()))
        classifier.show_most_informative_features()
        classifier.classify(word_feats(extract_feature(input_sentence)))
