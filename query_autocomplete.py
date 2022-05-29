# importing required dependencies
import pandas as pd
import math
import nltk
from nltk.util import ngrams
from english_words import english_words_set
import operator
import logging
import pickle
import os


# logging configurations
log_format = "[%(name)s][%(levelname)-6s] %(message)s"
logging.basicConfig(format=log_format)
query_autocomplete_logger = logging.getLogger("query_autocomplete")
query_autocomplete_logger.setLevel(logging.DEBUG)


class AutocompleteQuery():
    
    def __init__(self, sents_list, dicts_pth=''):
        '''
        arguments: sents_list(list), list of unique sentences
        '''
        
        # creating word distribution and n-gram dictionaries
        if dicts_pth == '':
            # creating words distibution
            self.create_wrd_dist(sents_list)
        
            # creating n-gram dictionary
            self.create_ngram_dict(sents_list)
        
        else:
            # loading all dictionaries from dicts_pth
            self.load_dicts(dicts_pth)
        
        
        
    def create_wrd_dist(self, sents_list):
        '''
        Creating the word distribution from sentences list
        
        Arguments: sents_list(list), list of unique sents
        '''
        
        query_autocomplete_logger.info(f"Creating word distribution. {len(sents_list)} sentences to be processed.")
        # creating word distribution dictionary
        self.word_dist_dict = {}   # {word: occurence count}
        for key in sents_list:
            wrd_list = nltk.word_tokenize(key)
            
            for wrd in wrd_list:
                if (wrd.title() in english_words_set) or (wrd.lower() in english_words_set):
                    # if word is present in english word data
                    wrd = wrd.lower()
                    if wrd not in self.word_dist_dict.keys():
                        self.word_dist_dict[wrd] = 1
                    else:
                        self.word_dist_dict[wrd] += 1
        query_autocomplete_logger.info(f"Finished creating word distribution.")
        
    
    def possible_words(self, initials='', psbl_wrd_cnt=3, return_all=False):
        '''
        Returns top multiple(psbl_wrd_cnt) possible words for incomplete word
        
        Args: initials(str), incomplete word i.e. to be completed
              psbl_wrds_cnt(int), count of possible words to return
              
        Return: list of possible words
        '''
        
        initials = initials.lower()
        
        # extracting all words starting with given initials
        all_psbl_wrds = {}    # {word: count of occurences}
        for key, value in self.word_dist_dict.items():
            if key.startswith(initials):
                all_psbl_wrds[key] = value
            else:
                pass
        
        # ranking all words as per occurence
        sorted_psbl_wrds = sorted(all_psbl_wrds.items(), key=operator.itemgetter(1), reverse=True)
        sorted_psbl_wrds = dict(sorted_psbl_wrds)  # converting tuples into dictionary
        
        # choosing possible words as per required count
        sorted_psbl_wrds = list(sorted_psbl_wrds.keys())
        
        psbl_wrds_cnt = min(len(sorted_psbl_wrds), psbl_wrd_cnt)
        if return_all == False:
            psbl_wrds = list(sorted_psbl_wrds)[:psbl_wrd_cnt]
        else:
            # return all possible words
            psbl_wrds = list(sorted_psbl_wrds)
            
        return psbl_wrds
    
    
    def sort_ngram_dict(ngram_dict):
        '''
        Sorting the value of the dictionary as per occurence count value
        
        Args: ngram_dict(dict), {word: {next_wrd: occurence count, ...}, ...}
        
        Returns: Returns sorted dictionary
        '''
        
        for context_wrd in ngram_dict.keys():
            ngram_dict[context_wrd] = dict(sorted(ngram_dict[context_wrd].items(), key=operator.itemgetter(1), reverse=True))
    
        return ngram_dict
    
    
    def create_ngram_dict(self, sents_list):
        '''
        creating bi-gram, tri-gram and quad-gram dict using sentences corpus
        
        Arguments: sents_list(list), list of unique sents
        
        '''
        
        query_autocomplete_logger.info(f"Creating Bi-grams. {len(sents_list)} sentences to be processed.")
        # creating bigram dictionary
        bigram_dict = {}    # {word: {next_wrd: occurence count, ...}, ...}
        for sent in sents_list:
            
            # creating bigrams from sentence
            bigrams = ngrams(nltk.word_tokenize(sent.lower()), 2)
            
            for item in bigrams:
                if item[0] in bigram_dict:
                    if item[1] in bigram_dict[item[0]]:
                        bigram_dict[item[0]][item[1]] += 1
                    else:
                        bigram_dict[item[0]][item[1]] = 1
                else:
                    bigram_dict[item[0]] = {item[1]: 1}
            
        # sorting the N-gram dictionary
        bigram_dict = sort_ngram_dict(bigram_dict)
            
        self.bigram_dict = bigram_dict
        query_autocomplete_logger.info(f"Finished creating Bi-grams.")
        
        query_autocomplete_logger.info(f"Creating Tri-grams. {len(sents_list)} sentences to be processed.")
        # creating tri-gram dictionary
        trigram_dict = {}
        for sent in sents_list:
            
            ngram = 3
            # creating trigram from sentences 
            trigrams = ngrams(nltk.word_tokenize(sent.lower()), ngram)
            
            for item in trigrams:
                if item[:ngram-1] in trigram_dict:
                    if item[ngram-1] in trigram_dict[item[:ngram-1]]:
                        trigram_dict[item[:ngram-1]][item[ngram-1]] += 1
                    else:
                        trigram_dict[item[:ngram-1]][item[ngram-1]] = 1
                else:
                    trigram_dict[item[:ngram-1]] = {item[ngram-1]: 1}
                    
        # sorting the N-gram dictionary
        trigram_dict = sort_ngram_dict(trigram_dict)    
            
        self.trigram_dict = trigram_dict
        query_autocomplete_logger.info(f"Finished creating Tri-grams.")
        
        query_autocomplete_logger.info(f"Creating Quad-grams. {len(sents_list)} sentences to be processed.")
        # creating quad-gram dictionary
        quadgram_dict = {}
        for sent in sents_list:
            
            ngram = 4
            # creating trigram from sentences 
            quadgrams = ngrams(nltk.word_tokenize(sent.lower()), ngram)
            
            for item in quadgrams:
                if item[:ngram-1] in quadgram_dict:
                    if item[ngram-1] in quadgram_dict[item[:ngram-1]]:
                        quadgram_dict[item[:ngram-1]][item[ngram-1]] += 1
                    else:
                        quadgram_dict[item[:ngram-1]][item[ngram-1]] = 1
                else:
                    quadgram_dict[item[:ngram-1]] = {item[ngram-1]: 1}
                    
        # sorting the N-gram dictionary
        quadgram_dict = sort_ngram_dict(quadgram_dict)    
            
        self.quadgram_dict = quadgram_dict
        query_autocomplete_logger.info(f"Finished creating Quad-grams.")
        
        return 1 
    
    
    def save_dicts(self, save_pth='./data/autocomplete_query'):
        '''
        save words distribution and n-grams dictionaries
        
        Arguments: save_pth (str), path were dictionaries will be saved
        '''    
    
        # create path if does not exist
        if not os.path.exists(save_pth):
            os.mkdir(save_pth)
            
        with open(save_pth+'/'+'wrds_dist_dict.pickle', 'wb') as handle:
            pickle.dump(self.word_dist_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(save_pth+'/'+'bigrams_dict.pickle', 'wb') as handle:
            pickle.dump(self.bigram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(save_pth+'/'+'trigrams_dict.pickle', 'wb') as handle:
            pickle.dump(self.trigram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(save_pth+'/'+'quadgrams_dict.pickle', 'wb') as handle:
            pickle.dump(self.quadgram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        query_autocomplete_logger.info(f"Dictionaries are saved to {save_pth}")
        
        return 1
    
    
    def load_dicts(self, dicts_pth):
        '''
        loads words distribution and n-grams dictionaries
        
        Arguments: dicts_pth(str), path from where data is going to be loaded
        '''
        
        with open(dicts_pth+'/'+'wrds_dist_dict.pickle', 'rb') as handle:
            self.word_dist_dict = pickle.load(handle)
            
        with open(dicts_pth+'/'+'bigrams_dict.pickle', 'rb') as handle:
            self.bigram_dict = pickle.load(handle)
        
        with open(dicts_pth+'/'+'trigrams_dict.pickle', 'rb') as handle:
            self.trigram_dict = pickle.load(handle)
            
        with open(dicts_pth+'/'+'quadgrams_dict.pickle', 'rb') as handle:
            self.quadgram_dict = pickle.load(handle)
        
        query_autocomplete_logger.info(f"All dictionaries are loaded from {dicts_pth}")
        
        return 1
        
        
    def predict_next_word(self, context, psbl_wrd_cnt=3):
        '''
        Predicting next word given the context word
        
        Args: context(str), incomplete sentence
        
        Returns: list of next possible words
        '''
        
        context_words = nltk.word_tokenize(context)  # list of words
        
        if len(context_words) >= 3:
            # using quadgram dictionary
            if (context_words[-3], context_words[-2], context_words[-1]) in self.quadgram_dict.keys():
                psbl_wrds = list(self.quadgram_dict[(context_words[-3], context_words[-2], context_words[-1])].keys())[:psbl_wrd_cnt]
                #print("trigram dict")
                return psbl_wrds
        
        if len(context_words) >= 2:
            # using trigram dictionary
            if (context_words[-2], context_words[-1]) in self.trigram_dict.keys():
                psbl_wrds = list(self.trigram_dict[(context_words[-2], context_words[-1])].keys())[:psbl_wrd_cnt]
                #print("trigram dict")
                return psbl_wrds
        
        if len(context_words) >= 1:
            # using bigram dictionary 
            if context_words[-1] in self.bigram_dict.keys():
                psbl_wrds = list(self.bigram_dict[context_words[-1]].keys())[:psbl_wrd_cnt]
                #print("bigram dict")
                return psbl_wrds
            
        return []
    
    
    def complete_last_word(self, sentence, psbl_wrd_cnt=3):
        '''
        returns the updated sentence with completed last word
        
        Args: sentence(str), incomplete query
              psbl_wrd_cnt(int), count of suggestions for last wrd to be returned
        
        Returns: list, list of suggested last words
        '''
        
        context_words = nltk.word_tokenize(sentence)  # list of words
        last_word = context_words[-1]
        context_words = context_words[:-1]
        
        if len(context_words) >= 3:
            # using quadgram dictionary
            
            if (context_words[-3], context_words[-2], context_words[-1]) in self.quadgram_dict.keys():
                all_psbl_wrds = list(self.quadgram_dict[(context_words[-3], context_words[-2], context_words[-1])].keys())
                
                # filtering words starting with last words
                filtr_wrds = []
                for wrd in all_psbl_wrds:
                    if wrd.startswith(last_word):
                        filtr_wrds.append(wrd)
                
                psbl_wrds = filtr_wrds[:psbl_wrd_cnt]
                if psbl_wrds != []:
                    return psbl_wrds
                
        
        if len(context_words) >= 2:
            # using trigram dictionary
            
            if (context_words[-2], context_words[-1]) in self.trigram_dict.keys():
                all_psbl_wrds = list(self.trigram_dict[(context_words[-2], context_words[-1])].keys())
            
                # filtering words starting with last words
                filtr_wrds = []
                for wrd in all_psbl_wrds:
                    if wrd.startswith(last_word):
                        filtr_wrds.append(wrd)
                
                psbl_wrds = filtr_wrds[:psbl_wrd_cnt]
                if psbl_wrds != []:
                    return psbl_wrds
        
        if len(context_words) >= 1:
            # using bigram dictionary 
            
            if context_words[-1] in self.bigram_dict.keys():
                all_psbl_wrds = list(self.bigram_dict[context_words[-1]].keys())
                
                # filtering words starting with last words
                filtr_wrds = []
                for wrd in all_psbl_wrds:
                    if wrd.startswith(last_word):
                        filtr_wrds.append(wrd)
                
                psbl_wrds = filtr_wrds[:psbl_wrd_cnt]
                if psbl_wrds != []:
                    return psbl_wrds
        
        # not using context for predicting last word if none of the ngram is used
        return self.possible_words(initials=last_word, psbl_wrd_cnt=psbl_wrd_cnt, return_all=False)
    
    
    def autocomplete_query(self, sentence, branches=2, levels=2):
        '''
        predict next n numbers of words
        
        Args: sentence(str), 
              branches(int),
              levels(int), 
        
        Returns: list of strings, sentence with predictions
        '''
        
        if levels <= 0:
            # no predictions will be generated
            return []
        
        # sentnece lower case
        sentence = sentence.lower()
        
        # word tokenization
        tokenized_wrds = nltk.word_tokenize(sentence)
        
        # storing all possible sentences
        psbl_sents = []
        
        # checking if last word is correct
        
        if sentence.endswith(' '):
            # we'll assume the last word is correct even if its not present in our data
            # predicting further possible sentences
            next_wrds = self.predict_next_word(sentence, psbl_wrd_cnt=branches)
            for wrd in next_wrds:
                psbl_sents.append(sentence + wrd)
            #print("ending with ' '", next_wrds)
        
        else:
            # sentence not ending with ' '(space)
            
            if tokenized_wrds[-1] in self.word_dist_dict.keys():
                # if last word present in word_dist_dict
                # predicting further possible sentences
                next_wrds = self.predict_next_word(sentence, psbl_wrd_cnt=branches)
                for wrd in next_wrds:
                    psbl_sents.append(sentence + ' ' + wrd)
                #print("not ending with ' '", next_wrds)
                
            else:
                # if last word is not present find the possible words
                psbl_last_wrds = self.complete_last_word(sentence=sentence, psbl_wrd_cnt=branches)
                for wrd in psbl_last_wrds:
                    psbl_sents.append(sentence[:-len(tokenized_wrds[-1])]+wrd)
                
        # predicting further querie from psbl_sents, as per branches and levels
        current_batch = psbl_sents
    
        for i in range(levels-1):
            
            next_batch = []
            for sentence in current_batch:
                next_wrds = self.predict_next_word(sentence, psbl_wrd_cnt=branches)
                for wrd in next_wrds:
                    next_batch.append(sentence + ' ' + wrd)
            
            psbl_sents.extend(next_batch)
            current_batch = next_batch
        
        
        return psbl_sents

