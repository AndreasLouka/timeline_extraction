import utils.utils as utils
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import CountVectorizer
import utils.cat_parser as cat_parser
import copy
import subprocess
import os
import numpy as np
import itertools
import cdeo_config
import nltk
from nltk import word_tokenize
import numpy as np
from itertools import chain
import collections
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import nltk.stem.porter as porter
from practnlptools.tools import Annotator
import EntityExtraction


reload(utils)
#np.set_printoptions(threshold=np.nan)

# init stemmer
stemmer = porter.PorterStemmer()

# init practnlp annotator for Semantic Role Labelling (SRL)
annotator = Annotator()

def getGoldEventMentions(ncorpus, targetEntityList):
	corpus = ncorpus
	GoldEvents = []
	goldTimelineDir = '../data/evaluation/corpus_%s/TimeLines_Gold_Standard' %(corpus)
	GoldEventMentions = (utils.goldTimelineList(targetEntityList, goldTimelineDir)) #list: (docId, sen, event, targetEntity, tmpDate, tmpOrder)

	return GoldEventMentions


def getFeaturesTrain(ncorpus, targetEntityList, GoldEventMentions, collection):

    corpus = ncorpus
    list_of_dict, labels = extractTrainingDictionaries(collection, GoldEventMentions, targetEntityList)

    v = DictVectorizer(sparse = False)
    F = v.fit_transform(list_of_dict)
    
    return list_of_dict, F, v, labels


def extractTrainingDictionaries (collection, GoldEvents, targetEntityList):

    list_of_dict = []
    labels = []

    for indoc in collection:
        doc = copy.deepcopy(indoc)
        doc_id = doc.get_doc_id()
     
        #SRL:
        #get targetEntities using Stanford CoreNLP:
        doc = EntityExtraction.getEntitiesStanfordNLP(indoc, targetEntityList)

        entity_list = list()
        for entity in doc.Markables.ENTITY_MENTION:
            entity_list.append((utils.getEntityText(doc, entity)).lower())
        entity_list = list(set(entity_list))


        for t_id,token in enumerate(doc.token):
            t = token.get_valueOf_()

            features_dict = collections.OrderedDict()

            #BAG-OF-WORDS:
            #words:
            if t in features_dict.keys():
                features_dict[t] += 1
            else:
                features_dict[t] = 1
            #2-grams:
            if (t_id) == 0:
                word_previous = 'start1'+'_'+t
            else:
                word_previous = doc.token[t_id-1].get_valueOf_()+'_'+t
            if word_previous in features_dict.keys():
                features_dict[word_previous] += 1
            else:
                features_dict[word_previous] = 1
            #3-grams:
            if (t_id) == 0:
                w_pre_previous = 'start2'+'_'+'start1'+t
            elif (t_id) == 1:
                w_pre_previous = 'start1'+'_'+doc.token[t_id-1].get_valueOf_()+'_'+t
            else:
                w_pre_previous = doc.token[t_id-2].get_valueOf_()+'_'+doc.token[t_id-1].get_valueOf_()+'_'+t
            if w_pre_previous in features_dict.keys():
                features_dict[w_pre_previous] += 1
            else:
                features_dict[w_pre_previous] =1

            
            #POS_TAGS
            tag = nltk.pos_tag([t])
            #tags:
            word_tag = tag[0][0]+'_'+tag[0][1]
            if word_tag in features_dict.keys():
                features_dict[word_tag] += 1
            else:
                features_dict[word_tag] = 1
            #Pos_tag 2-grams:
            if (t_id) == 0:
                previous_tag = 'start1'
                pos_pos = previous_tag+'_'+tag[0][0]
            else:
                previous_tag = nltk.pos_tag([doc.token[t_id - 1].get_valueOf_()])
                pos_pos = previous_tag[0][0]+'_'+tag[0][0]
            if pos_pos in features_dict.keys():
                features_dict[pos_pos] += 1
            else:
                features_dict[pos_pos] = 1
            #Pos_tag 3-grams:
            if (t_id) == 0:
                previous_tag_tag = 'start2'
                previous_tag = 'start1'
                pos_pos_pos = previous_tag_tag+'_'+previous_tag+'_'+tag[0][0]
            elif (t_id) == 1:
                previous_tag_tag = 'start1'
                previous_tag = nltk.pos_tag([doc.token[t_id - 1].get_valueOf_()])
                pos_pos_pos = previous_tag_tag+'_'+previous_tag[0][0]+'_'+tag[0][0]
            else:
                previous_tag_tag = nltk.pos_tag([doc.token[t_id - 2].get_valueOf_()])
                previous_tag = nltk.pos_tag([doc.token[t_id - 1].get_valueOf_()])
                pos_pos_pos = previous_tag_tag[0][0]+'_'+previous_tag[0][0]+'_'+tag[0][0]

            
            #Get sentence of token to extract SRL features:
            if t_id == 0:
                sentence = getSentence(doc, token)
                srl, verbs = getSRLfeatures(sentence, t)
            
            if t_id != 0:
                if (doc.token[t_id].sentence) != (doc.token[t_id - 1].sentence):
                    sentence = getSentence(doc, token)
                    srl, verbs = getSRLfeatures(sentence, t)

            if t in verbs:
                if 'verb' in features_dict.keys():
                    features_dict['verb'] += 1
                else:
                    features_dict['verb'] = 1

            for dictionary in srl:
                if ('A0' in dictionary.keys()) and ('V' in dictionary.keys()):
                    if (dictionary['A0'].lower().split(' ')[0] in entity_list) and (dictionary['V'] == t):
                        if 'A0_verb' in features_dict.keys():
                            features_dict['A0_verb'] += 1
                        else:
                            features_dict['A0_verb'] =1
                if ('A1' in dictionary.keys()) and ('V' in dictionary.keys()):
                    if (dictionary['A1'].lower().split(' ')[0] in entity_list) and (dictionary['V'] == t):
                        if 'A1_verb' in features_dict.keys():
                            features_dict['A1_verb'] += 1
                        else:
                            features_dict['A1_verb'] = 1
            
            list_of_dict.append(features_dict)

            #labelling:
            #match = [item for item in GoldEvents if (item[0], item[1], item[2]) == (doc_id, token.sentence, t)]
            match = [item for item in GoldEvents if (item[2]) == (t)]

            if not match:
                labels.append(0)
            else:
                labels.append(1)

    return list_of_dict, labels


def getTrainedClassifier (features, labels):

    n_iter = cdeo_config.getConfig('n_epochs_entity')
    clf = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, n_iter=n_iter, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False)
    #clf = LogisticRegression(solver='liblinear', C=0.5, penalty='l2', tol=1e-5, class_weight='auto', fit_intercept=True)
    clf.fit (features, labels)

    return clf  


def getSentence(document,token):
    sentence = ''
    for tok in document.token:
        if token.sentence == tok.sentence:
            sentence = sentence + tok.get_valueOf_() + ' '
    return sentence    


def getSRLfeatures(sentence, token):
    
    s = sentence.encode('utf-8')
    s = s.translate(None, '?')
    annotation = annotator.getAnnotations(s, dep_parse = True)
    srl = annotation['srl']
    ner = annotation['ner']
    verbs = annotation['verbs']

    return srl, verbs


def getFeaturesTest (indoc, targetEntityList, clf, vectorizer):

    m_id = 0
    doc = copy.deepcopy(indoc)
    doc.Markables.EVENT_MENTION = [] # remove any previous event annotations
    txt = utils.getRawText(doc).split(' ')
    #print doc.get_doc_id()


    raw_doc = utils.getRawTextLines(doc)

    split_doc = raw_doc.split('\n')

    candidate_ids = list()

    #Entity_list (using stanford CoreNLP)
    entity_list = list()
    for entity in doc.Markables.ENTITY_MENTION:
        entity_list.append((utils.getEntityText(doc, entity)).lower())
    entity_list = list(set(entity_list))

    for t_id,token in enumerate(doc.token):
        t = token.get_valueOf_()

        features_dict = collections.OrderedDict()


        #BAG-OF-WORDS:
        #words:
        if t in features_dict.keys():
            features_dict[t] += 1
        else:
            features_dict[t] =1
        #2-grams:
        if (t_id) == 0:
            word_previous = 'start1'+'_'+t
        else:
            word_previous = doc.token[t_id-1].get_valueOf_()+'_'+t
        if word_previous in features_dict.keys():
            features_dict[word_previous] += 1
        else:
            features_dict[word_previous] = 1
        #3-grams:
        if (t_id) == 0:
            w_pre_previous = 'start2'+'_'+'start1'+t
        elif (t_id) == 1:
            w_pre_previous = 'start1'+'_'+doc.token[t_id-1].get_valueOf_()+'_'+t
        else:
            w_pre_previous = doc.token[t_id-2].get_valueOf_()+'_'+doc.token[t_id-1].get_valueOf_()+'_'+t
        if w_pre_previous in features_dict.keys():
            features_dict[w_pre_previous] += 1
        else:
            features_dict[w_pre_previous] =1
        


        #POS_TAGS
        tag = nltk.pos_tag([t])
        #tags:
        word_tag = tag[0][0]+'_'+tag[0][1]
        if word_tag in features_dict.keys():
            features_dict[word_tag] += 1
        else:
            features_dict[word_tag] = 1
        #Pos_tag 2-grams:
        if (t_id) == 0:
            previous_tag = 'start1'
            pos_pos = previous_tag+'_'+tag[0][0]
        else:
            previous_tag = nltk.pos_tag([doc.token[t_id - 1].get_valueOf_()])
            pos_pos = previous_tag[0][0]+'_'+tag[0][0]
        if pos_pos in features_dict.keys():
            features_dict[pos_pos] += 1
        else:
            features_dict[pos_pos] = 1
        #Pos_tag 3-grams:
        if (t_id) == 0:
            previous_tag_tag = 'start2'
            previous_tag = 'start1'
            pos_pos_pos = previous_tag_tag+'_'+previous_tag+'_'+tag[0][0]
        elif (t_id) == 1:
            previous_tag_tag = 'start1'
            previous_tag = nltk.pos_tag([doc.token[t_id - 1].get_valueOf_()])
            pos_pos_pos = previous_tag_tag+'_'+previous_tag[0][0]+'_'+tag[0][0]
        else:
            previous_tag_tag = nltk.pos_tag([doc.token[t_id - 2].get_valueOf_()])
            previous_tag = nltk.pos_tag([doc.token[t_id - 1].get_valueOf_()])
            pos_pos_pos = previous_tag_tag[0][0]+'_'+previous_tag[0][0]+'_'+tag[0][0]

        
        #Get sentence of token to extract SRL features:
        if t_id == 0:
            sentence = getSentence(doc, token)
            srl, verbs = getSRLfeatures(sentence, t)
        
        if t_id != 0:
            if (doc.token[t_id].sentence) != (doc.token[t_id - 1].sentence):
                sentence = getSentence(doc, token)
                srl, verbs = getSRLfeatures(sentence, t)

        if t in verbs:
            if 'verb' in features_dict.keys():
                features_dict['verb'] += 1
            else:
                features_dict['verb'] = 1

        for dictionary in srl:
            if ('A0' in dictionary.keys()) and ('V' in dictionary.keys()):
                if (dictionary['A0'].lower().split(' ')[0] in entity_list) and (dictionary['V'] == t):
                    if 'A0_verb' in features_dict.keys():
                        features_dict['A0_verb'] += 1
                    else:
                        features_dict['A0_verb'] =1
            if ('A1' in dictionary.keys()) and ('V' in dictionary.keys()):
                if (dictionary['A1'].lower().split(' ')[0] in entity_list) and (dictionary['V'] == t):
                    if 'A1_verb' in features_dict.keys():
                        features_dict['A1_verb'] += 1
                    else:
                        features_dict['A1_verb'] = 1
         

        #Vectorize:
        F = vectorizer.transform(features_dict)

        predicted_events = clf.predict(F)

        if predicted_events == 1:
            candidate_ids += [[t_id+1]]

        for dictionary in srl:
            if ('A0' in dictionary.keys()):
                if (dictionary['A0'].lower().split(' ')[0] in entity_list):
                    if ('AM-MOD' in dictionary.keys()):
                        if dictionary['AM-MOD'].lower().split(' ')[0] == 'will':
                            if t in verbs:
                                candidate_ids += [[t_id + 1]]
                    else:
                        if t in verbs:
                            candidate_ids += [[t_id + 1]]

            if ('A1' in dictionary.keys()):
                if (dictionary['A1'].lower().split(' ')[0] in entity_list):
                    if ('AM-MOD' in dictionary.keys()):
                        if dictionary['AM-MOD'].lower().split(' ')[0] == 'will':
                            if t in verbs:
                                candidate_ids += [[t_id + 1]]
                    else:
                        if t in verbs:
                            candidate_ids += [[t_id + 1]]

        if t in verbs:
            candidate_ids += [[t_id + 1]]
        
        new_candidate_ids = list()
        for elem in candidate_ids:
            if elem not in new_candidate_ids:
                new_candidate_ids.append(elem)
        

    m_id = 0
    for t in new_candidate_ids:
        # prep the token id list
        token_anchor_list = list()
        for t0 in t:
            tmp = cat_parser.token_anchorType1()
            tmp.set_t_id(t0)
            token_anchor_list.append(tmp)

            # build the EVENT_MENTION structure as in cat_parser
            event = cat_parser.EVENT_MENTIONType()
            event.set_token_anchor(token_anchor_list) 
            event.set_m_id(m_id)
            m_id+=1
            doc.Markables.EVENT_MENTION.append(event)
         
    return doc



