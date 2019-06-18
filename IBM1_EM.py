#!/usr/bin/env python
# coding: utf-8

# In[196]:


import nltk
import numpy as np
import pandas as pd
import json


# In[251]:


print("Give the file name of the parallel corpus")
file_name=raw_input()


# In[259]:


#Load the data and make proper sentences
#fr=[], will contain a list of French sentences
#en=[], will contain a list of English sentences

json_data=open('data3.json').read()
data = json.loads(json_data)
#print(type(data))
fr=[]
en=[]
for i in data:
    fr.append(i["fr"])
    en.append(i["en"])


# In[260]:


#Each sentence is tokenized into a list for ease of processing
for i in range(len(fr)):
    fr[i]=str(fr[i])
    en[i]=str(en[i])

for i in range(len(fr)):
    fr[i]=fr[i].split(' ')
    en[i]=en[i].split(' ')


# In[261]:


#distinct_en will contain a list of distinct English words
#distinct_fr will contain a list of distinct French words

distinct_en=[]
distinct_fr=[]

for e in en:
    for word in e:
        if(word not in distinct_en):
            distinct_en.append(word)

for f in fr:
    for word in f:
        if(word not in distinct_fr):
            distinct_fr.append(word)


# In[262]:


#Debug
print("Distinct words collected")
print(" ")
print(distinct_en)
print(distinct_fr)


# In[263]:


#t_val is a dictionary where each key is of the form (english_word, french_word) and the corresponding value
#is t(english_word|french_word)
#We have to initialize them uniformly, so t(english_word|french_word) for each pair becomes 1/(number of distinct_english_words)

t_val={}
for f_word in distinct_fr:
    for e_word in distinct_en:
            t_val[(e_word,f_word)]=1.0/(len(distinct_en))
            
#Pairs and initial probabilities
'''
for key,val in sorted(t_val.items()):
    print(key,val)
'''

# In[264]:


#TRAINING MY MODEL
#We run the model for 100 iterations or until it converges, whichever is earlier
#epoch = number of iterations

epoch=10000
while(epoch!=0):
    
    total_difference=0.0
    #counts of English word given a French Word
    count={}
    
    total={}
    
    for f_word in distinct_fr:
        for e_word in distinct_en:
            count[(e_word,f_word)]=0
    for f_word in distinct_fr:
        total[f_word]=0
    
    for i in range(len(fr)):
        f=fr[i]
        e=en[i]
        #normalization for an English word weighed by the prior translation probabilities
        s_total={}
        #stores the sum of translation probabilities corresponding to a French word in the sentence, normalized
        for e_word in e:
            s_total[e_word]=0
            for f_word in f:
                s_total[e_word]+=(t_val[(e_word,f_word)])
            #endfor
        #endfor

        for e_word in e:
            for f_word in f:
                count[(e_word,f_word)]+=(t_val[(e_word,f_word)]/s_total[e_word])
                total[f_word]+=(t_val[(e_word,f_word)]/s_total[e_word])
                #print(f_word, total[f_word])
            #endfor
        #endfor
    #endfor_twice
    #print(total)
    
    for f_word in distinct_fr:
        for e_word in distinct_en:
            total_difference+=abs((t_val[(e_word,f_word)]-(count[(e_word,f_word)]/total[f_word])))
            t_val[(e_word,f_word)]=(count[(e_word,f_word)]/total[f_word])
        #endfor
    #endfor
    
    
    #print("Total displacement : ", total_difference)
    if(total_difference < 0.0001):
        break
    epoch-=1
print(" ")
print("Number of iterations to convergence" , 10000-epoch)
print(" ")

# In[265]:


#ALIGNMENTS OF MY MODEL
print("Implemented MODEL")
for i in range(len(fr)):
    f=np.array(fr[i])#French sentence in the pair
    e=np.array(en[i])#English sentence in the pair
    #print(e)
    #print(f)
    align=[]
    #For every word in the English sentence, check which word in the French sentence has the maximum
    #probability of producing the former and align them
    for e_i in range(len(e)):
        max_sim=-1
        max_f=-1
        for f_i in range(len(f)):
            if(t_val[(e[e_i],f[f_i])]>max_sim):
                max_f=f_i
                max_sim=t_val[(e[e_i],f[f_i])]
        align.append((e_i,max_f))
    print(align)

print(" ")
# In[266]:


#TRAINING IBM MODEL 1
from collections import defaultdict
from nltk.translate import AlignedSent
from nltk.translate import Alignment
from nltk.translate import IBMModel,IBMModel1, IBMModel2
from nltk.translate.ibm_model import Counts
#bitext will have the parallel corpus
bitext=[]
for i in range(len(fr)):
    bitext.append(AlignedSent(en[i],fr[i]))
#Training for 100 iterations
ibm1 = IBMModel1(bitext,1000)
#trans_dict will contain the translation probabilities for each distinct pair of words
#pair being of the form (english_word,french_word)
trans_dict=ibm1.translation_table


# In[267]:


#ALIGNMENTS OF IBM MODEL 1

print("IBM MODEL 1")
for i in range(len(fr)):
    test_sentence=bitext[i]
    align_ibm=test_sentence.alignment
    #print(test_sentence)
    print(align_ibm)
    #print(" ")
print(" ")

# In[268]:


#TRAINING IBM MODEL 2
#bitext_2 will have the parallel corpus

bitext_2=[]
for i in range(len(fr)):
    bitext_2.append(AlignedSent(en[i],fr[i]))
    
#Training for 100 iterations    
ibm2 = IBMModel2(bitext_2,1000)
#trans_dict_2 will contain the translation probabilities for each distinct pair of words
#pair being of the form (english_word,french_word)
trans_dict_2=ibm2.translation_table


# In[269]:


#ALIGNMENTS OF IBM MODEL 2
print("IBM MODEL 2")
for i in range(len(fr)):
    test_sentence=bitext_2[i]
    align_ibm2=test_sentence.alignment
    #print(test_sentence)
    print(align_ibm2)
    #print(" ")
print(" ")

# In[270]:


'''Here then notion of source and target gets reversed
The source is basically the source of alignment - which is actually the target for IBM1 i.e English'''
from nltk.translate.phrase_based import phrase_extraction
count_fr_phrase={}
count_en_fr_phrase={}
for i in range(len(fr)):
    
    test_sentence = bitext[i]
    #print(test_sentence)
    align_ibm=test_sentence.alignment
    
    
    f=np.array(fr[i])
    e=np.array(en[i])
    align=[]
    #For each sentence pair, make the alignment
    for e_i in range(len(e)):
        max_sim=-1
        max_f=-1
        for f_i in range(len(f)):
            if(t_val[(e[e_i],f[f_i])]>max_sim):
                max_f=f_i
                max_sim=t_val[(e[e_i],f[f_i])]
        align.append((e_i,max_f))

    #Construct the source and target texts
    srctext=""
    trgtext=""
    for e_word in e:
        srctext+=e_word
        srctext+=' '
    for f_word in f:
        trgtext+=f_word
        trgtext+=' '
    srctext=srctext[:-1]
    trgtext=trgtext[:-1]
    #print(srctext)
    #print(trgtext)
    
    #Obtain phrase tuples from phrase_extraction module
    phrases = phrase_extraction(srctext, trgtext, align_ibm)
    for phrase in sorted(phrases):
        en_phrase=phrase[2]#English phrase
        fr_phrase=phrase[3]#French phrase
        #Increment count of the French phrase 
        if(fr_phrase not in count_fr_phrase):
            count_fr_phrase[fr_phrase]=1
        else:
            count_fr_phrase[fr_phrase]+=1
        
        #Increment count of the pair of English phrase, French phrase
        if((en_phrase,fr_phrase) not in count_en_fr_phrase):            
            count_en_fr_phrase[(en_phrase,fr_phrase)]=1
        else:            
            count_en_fr_phrase[(en_phrase,fr_phrase)]+=1    
                
print(' ')


# In[271]:

print("Phrase translation scores in descending order")
print(" ")
#This list will contain tuples of the form (phrase_translation_score,(english_phrase, foreign_phrase))
score_to_phrase_pair=[]
phrase_t={}
for key,val in sorted(count_en_fr_phrase.items()):
    #print(key,val)
    phrase_t[key]=1.0*val/count_fr_phrase[key[1]]
    score_to_phrase_pair.append((phrase_t[key],key))

for i in sorted(score_to_phrase_pair)[::-1]:
    print(i)


# In[ ]:




