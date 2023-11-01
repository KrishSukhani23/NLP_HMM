import pandas as pd
import numpy as np
import sys

#No. of Arguments = 2

data_path = sys.argv[1]
# output_path = sys.argv[2]

data_csv = pd.read_csv(data_path + 'train', on_bad_lines='skip', sep='\t', low_memory=False, names=["Index", "Word_Type", "POS_Tag"])
data  = data_csv.groupby(["Word_Type"])["POS_Tag"].size().reset_index(name="Counts")
data = data.sort_values(by=['Counts'], ascending=False)
data['Word_Type'].mask(data['Counts'] <=3 ,'<unk>', inplace=True)
new_row = pd.DataFrame({'Word_Type' : '<unk>', 'Counts' : data['Word_Type'].value_counts()['<unk>']}, index=[0])
data = data[data.Word_Type != '<unk>']
new_df = pd.concat([new_row, data]).reset_index(drop = True)
new_df = new_df.reset_index()

new_df = new_df[['Word_Type', 'index', 'Counts']]
new_df.to_csv('vocab.txt', sep='\t', header=None, index=None)


"""#Task 2"""

transmission = {}
emission = {}

data  = data_csv.groupby(["Word_Type"])["POS_Tag"].size().reset_index(name="Counts")
unk_list = list(data[data['Counts']<=3]['Word_Type'])

data_csv['Word_Type'][data_csv['Word_Type'].isin(unk_list)==True] = '<unk>'

index = data_csv['Index'].values.tolist()
word_type = data_csv['Word_Type'].values.tolist()
pos_tag = data_csv['POS_Tag'].values.tolist()

count_pair = {}
count_new_senetences =  0
for i in range(len(pos_tag)-1):
    if index[i] == 1 and f'(INI, {pos_tag[i]})' in count_pair:
      count_pair[f'(INI, {pos_tag[i]})'] += 1
      count_new_senetences +=  1
    if index[i] == 1 and f'(INI, {pos_tag[i]})' not in count_pair:
      count_pair[f'(INI, {pos_tag[i]})'] = 1
    if index[i+1] == 1:
      continue
    if f'({pos_tag[i]},{pos_tag[i+1]})' in count_pair:
      count_pair[f'({pos_tag[i]},{pos_tag[i+1]})'] += 1
    else:
      count_pair[f'({pos_tag[i]},{pos_tag[i+1]})'] = 1

pos_tag_count = {}
for i in range(len(pos_tag)):
    if pos_tag[i] in pos_tag_count:
      pos_tag_count[pos_tag[i]] += 1
    else:
      pos_tag_count[pos_tag[i]] = 1

tag_to_word = {}
for i in range(len(pos_tag)):
    if f'({pos_tag[i]},{word_type[i]})' in tag_to_word:
      tag_to_word[f'({pos_tag[i]},{word_type[i]})'] += 1
    else:
      tag_to_word[f'({pos_tag[i]},{word_type[i]})'] = 1


for i in range(len(index)):
    if index[i] == 1:
      transmission[f'(INI, {pos_tag[i]})'] = count_pair[(f'(INI, {pos_tag[i]})')] / count_new_senetences
    # if i!=len(index) and index[i+1] == 1:
    #   continue
    if index[i-1]+1 == index[i]:
      transmission[f'({pos_tag[i-1]},{pos_tag[i]})'] = count_pair[f'({pos_tag[i-1]},{pos_tag[i]})'] / pos_tag_count[pos_tag[i-1]]
    # print(transmission)
    emission[f'({pos_tag[i]},{word_type[i]})'] = tag_to_word[f'({pos_tag[i]},{word_type[i]})'] / pos_tag_count[pos_tag[i]]

import json
with open('hmm.json', 'a', encoding="utf-8") as file:
    x = json.dumps(emission, indent=4)
    file.write(x + '\n')
with open('hmm.json', 'a', encoding="utf-8") as file:
    x = json.dumps(transmission, indent=4)
    file.write(x + '\n')

print("Number of transmission parameters: "+ str(len(transmission)))
print("Number of emission parameters: "+ str(len(emission)))

"""#Task 3

"""

data_dev = pd.read_csv(data_path + '/dev', on_bad_lines='skip', sep='\t', low_memory=False, names=["Index", "Word_Type", "POS_Tag"])

count_pair = {}
count_new_senetences =  0
for i in range(len(pos_tag)-1):
    if index[i] == 1 and ('INI',pos_tag[i]) in count_pair:
      count_pair[('INI', pos_tag[i])] += 1
      count_new_senetences +=  1
    if index[i] == 1 and ('INI', pos_tag[i]) not in count_pair:
      count_pair[('INI', pos_tag[i])] = 1
      count_new_senetences +=  1
    if index[i+1] == 1:
      continue
    if (pos_tag[i], pos_tag[i+1]) in count_pair:
      count_pair[(pos_tag[i],pos_tag[i+1])] += 1
    else:
      count_pair[(pos_tag[i],pos_tag[i+1])] = 1

pos_tag_count = {}
for i in range(len(pos_tag)):
    if pos_tag[i] in pos_tag_count:
      pos_tag_count[pos_tag[i]] += 1
    else:
      pos_tag_count[pos_tag[i]] = 1

tag_to_word = {}
for i in range(len(pos_tag)):
    if (pos_tag[i],word_type[i]) in tag_to_word:
      tag_to_word[(pos_tag[i],word_type[i])] += 1
    else:
      tag_to_word[(pos_tag[i],word_type[i])] = 1

for i in range(len(index)):
    if index[i] == 1:
      transmission[('INI', pos_tag[i])] = count_pair[('INI', pos_tag[i])] / count_new_senetences
    # if i!=len(index) and index[i+1] == 1:
    #   continue
    if index[i-1]+1 == index[i]:
      transmission[(pos_tag[i-1], pos_tag[i])] = count_pair[(pos_tag[i-1], pos_tag[i])] / pos_tag_count[pos_tag[i-1]]
    # print(transmission)
    emission[(pos_tag[i],word_type[i])] = tag_to_word[(pos_tag[i],word_type[i])] / pos_tag_count[pos_tag[i]]

index_dev = data_dev['Index'].values.tolist()
word_type_dev = data_dev['Word_Type'].values.tolist()
pos_tag_dev = data_dev['POS_Tag'].values.tolist()
predicted_tag = {}

def calc_emission_data(word):
  trans_check = []
  for j in emission:
    if j[1] == word:
      trans_check.append(j)
  if len(trans_check) == 0:
    for j in emission:
      if j[1] == '<unk>':
        trans_check.append(j)
  return trans_check

def calc_trans_data(prev_tag, list_emiss):
  trans_list = []
  for i in list_emiss:
    if (prev_tag,i[0]) in transmission:
      trans_list.append((prev_tag,i[0]))
  return trans_list

def max_prob_list(trans_list):
  list_of_probabilities = []
  for m in range(len(trans_list)):
    # if transmission[trans_list[m]]*emission[trans_check[m]] != []:
    list_of_probabilities.append(transmission[trans_list[m]]*emission[trans_check[m]])
    # else:
      # list_of_probabilities.append(0)
  if len(list_of_probabilities) == 0:
    return ','
  return trans_list[list_of_probabilities.index(max(list_of_probabilities))][1]

predicted_tag = {}

for i in range(len(index_dev)):
  # print(word_type_dev[i])
  if index_dev[i] == 1:
    trans_check = calc_emission_data(word_type_dev[i])
    trans_list = calc_trans_data('INI', trans_check)
    # print(trans_check)
    # print(trans_list)
    predicted_tag[i] = max_prob_list(trans_list)
  else:
    trans_check = calc_emission_data(word_type_dev[i])
    trans_list = calc_trans_data(predicted_tag[i-1], trans_check)
    # print(trans_check)
    # print(trans_list)
    predicted_tag[i] = max_prob_list(trans_list)


correct = 0
miss = 0
for i in range(len(predicted_tag)):
  if pos_tag_dev[i] == predicted_tag[i]:
    correct+=1
print("Greedy Accuracy")
print(correct/len(predicted_tag))


#Test Data
data_test = pd.read_csv(data_path + 'test', on_bad_lines='skip', sep='\t', low_memory=False, names=["Index", "Word_Type"])

index_test = data_test['Index'].values.tolist()
word_type_test = data_test['Word_Type'].values.tolist()
predicted_tag = {}

for i in range(len(index_test)):
  if index_test[i] == 1:
    # print(word_type_test[i])
    trans_check = calc_emission_data(word_type_test[i])
    trans_list = calc_trans_data('INI', trans_check)
    # print(trans_check)
    # print(trans_list)
    predicted_tag[i] = max_prob_list(trans_list)
  else:
    trans_check = calc_emission_data(word_type_test[i])
    trans_list = calc_trans_data(predicted_tag[i-1], trans_check)
    # print(trans_check)
    # print(trans_list)
    predicted_tag[i] = max_prob_list(trans_list)


predicted = []

for i in range(len(predicted_tag)):
  predicted.append(predicted_tag[i])

greedy_out = pd.DataFrame(
    {'Index': index_test, 'Word_Type': word_type_test, 'Predicted_POS': predicted
    })

greedy_out.to_csv('greedy.out', sep='\t', header=None, index=None)

# greedy_out.to_csv('greedy_out.text', sep='\t', header=None, index=None)

"""#Viterbi"""

actual_matched = 0
wrong_matched = 0

pos_tag_count['INI'] = count_new_senetences

pos_tag_index = dict()
index = 0

start_index = -1
end_index = -1

for key in pos_tag_count:
	if key == ".":
		end_index = index
	if key == 'INI':
		start_index = index
	pos_tag_index[index] = key
	index += 1

dev = open(data_path + 'dev','r')

words_list = list()
actual_tag = list()

data_lines = dev.readlines()
# print(data_lines)
# def calc_max(predict_tag_array):
#   for k in range(len(predict_tag_array)):
#     if predict_tag_array[k] == actual_tag[k]:
#       actual_matched += 1
#     else:
#       wrong_matched += 1

for each_line in data_lines:
  words = each_line[:-1].split("\t")
  #Unless this sentence goes till the end
  if words[0]!="":
    if words[1] in unk_list:
      words_list.append('<unk>')
    else: 
      words_list.append(words[1])
    actual_tag.append(words[2])
  
  #If we encounter a new line
  else:
    dp = [[-1 for _ in range(len(words_list))] for _ in range(len(pos_tag_count))]
    # print(dp)
    for i in range(len(pos_tag_count)):
      #the tag right now
      current_tag = pos_tag_index[i]
      transmission_attribute = ('INI', current_tag)
      transmission_probability = 0.0000000000001
      #calculating transmission
      if transmission_attribute in transmission:
        transmission_probability = transmission[transmission_attribute]
        # print(transmission_probability)

      emission_attribute = (current_tag , words_list[0])
      emission_probabiity = 0.0000000000001
      #calculating emission
      if emission_attribute in emission:
        emission_probabiity = emission[emission_attribute]
        # print(emission_probabiity)

      dp[i][0] = transmission_probability*emission_probabiity
      # print(dp)
    for word_index in range(1,len(words_list)):
      cur_word = words_list[word_index]

      for i in range(len(pos_tag_count)):
        current_tag = pos_tag_index[i]
        max_prob = 0
        emission_attribute = (current_tag , cur_word)
        emission_probabiity = 0.0000000000001
        if emission_attribute in emission:
          emission_probabiity = emission[emission_attribute]
          # print(emission_probabiity)

        for j in range(len(pos_tag_count)):
          prev_tag = pos_tag_index[j]

          transmission_attribute = (prev_tag, current_tag)
          transmission_probability = 0.0000000000001
          if transmission_attribute in transmission:
            transmission_probability = transmission[transmission_attribute]
            # print(transmission_probability)

          max_prob = max(max_prob, dp[j][word_index-1]*emission_probabiity*transmission_probability)

        dp[i][word_index] = max_prob
        # print(dp)

    # print(predict_tag_array)
    predict_tag_array = []

    column = len(words_list) - 1
    next_tag = -1
    max_prob = 0
    for i in range(len(pos_tag_count)):
      if max_prob < dp[i][column]:
        max_prob = dp[i][column]
        next_tag = i

    if next_tag == -1:
      next_tag = end_index

    predict_tag_array.append(pos_tag_index[next_tag])
    # print(predict_tag_array)
    for column in range(len(words_list)-2,-1,-1):
      next_word = words_list[column+1]
      max_prev_tag = start_index
      store_max_prob = max_prob
      diff = 1
      for i in range(len(pos_tag_count)):
        prev_tag = i
        cur_prob = dp[i][column]

        if cur_prob != 0:
          transmission_attribute = (pos_tag_index[prev_tag] , pos_tag_index[next_tag])
          transmission_probability = 0.0000000000001
          if transmission_attribute in transmission:
            transmission_probability = transmission[transmission_attribute]


          emission_attribute = (pos_tag_index[next_tag] , next_word)
          emission_probabiity = 0.0000000000001
          if emission_attribute in emission:
            emission_probabiity = emission[emission_attribute]

          if diff > abs(cur_prob*(transmission_probability*emission_probabiity) - max_prob):
            diff = abs(cur_prob*(transmission_probability*emission_probabiity) - max_prob)
            max_prev_tag = prev_tag
            store_max_prob = cur_prob

      next_tag = max_prev_tag
      max_prob = store_max_prob

      predict_tag_array.append(pos_tag_index[max_prev_tag])

    predict_tag_array.reverse()

    for k in range(len(predict_tag_array)):
      if predict_tag_array[k] == actual_tag[k]:
        actual_matched += 1
      else:
        wrong_matched += 1
    # predd.append(predict_tag_array)
    words_list = []
    actual_tag = []
    # print(predd)

print("Viterbi - Accuracy: ")
print(str(actual_matched/(actual_matched+wrong_matched)))

#TEST Data

actual_matched = 0
wrong_matched = 0

pos_tag_count['INI'] = count_new_senetences


pos_tag_index = dict()
index = 0

start_index = -1
end_index = -1

for key in pos_tag_count:
	if key == ".":
		end_index = index
	if key == 'INI':
		start_index = index
	pos_tag_index[index] = key
	index += 1

test = open(data_path + 'test','r')

words_list = list()
actual_tag = list()

data_lines = test.readlines()

def calc_max(predict_tag_array):
  for k in range(len(predict_tag_array)):
    if predict_tag_array[k] == actual_tag[k]:
      actual_matched += 1
    else:
      wrong_matched += 1

output = open('viterbi.out','w')

predd = []
for each_line in data_lines:
  words = each_line[:-1].split("\t")
  #Unless this sentence goes till the end
  if words[0]!="":
    if words[1] in unk_list:
      words_list.append('<unk>')
    else: 
      words_list.append(words[1])
  
  #If we encounter a new line
  else:
    dp = [[-1 for _ in range(len(words_list))] for _ in range(len(pos_tag_count))]
    # print(dp)
    for i in range(len(pos_tag_count)):
      #the tag right now
      current_tag = pos_tag_index[i]
      transmission_attribute = ('INI', current_tag)
      transmission_probability = 0.0000000000001
      #calculating transmission
      if transmission_attribute in transmission:
        transmission_probability = transmission[transmission_attribute]
        # print(transmission_probability)

      emission_attribute = (current_tag , words_list[0])
      emission_probabiity = 0.0000000000001
      #calculating emission
      if emission_attribute in emission:
        emission_probabiity = emission[emission_attribute]
        # print(emission_probabiity)

      dp[i][0] = transmission_probability*emission_probabiity
      # print(dp)
    for word_index in range(1,len(words_list)):
      cur_word = words_list[word_index]

      for i in range(len(pos_tag_count)):
        current_tag = pos_tag_index[i]
        max_prob = 0
        emission_attribute = (current_tag , cur_word)
        emission_probabiity = 0.0000000000001
        if emission_attribute in emission:
          emission_probabiity = emission[emission_attribute]
          # print(emission_probabiity)

        for j in range(len(pos_tag_count)):
          prev_tag = pos_tag_index[j]

          transmission_attribute = (prev_tag, current_tag)
          transmission_probability = 0.0000000000001
          if transmission_attribute in transmission:
            transmission_probability = transmission[transmission_attribute]
            # print(transmission_probability)

          max_prob = max(max_prob, dp[j][word_index-1]*emission_probabiity*transmission_probability)

        dp[i][word_index] = max_prob
        # print(dp)


    predict_tag_array = []

    column = len(words_list) - 1
    next_tag = -1
    max_prob = 0
    for i in range(len(pos_tag_count)):
      if max_prob < dp[i][column]:
        max_prob = dp[i][column]
        next_tag = i

    if next_tag == -1:
      next_tag = end_index

    predict_tag_array.append(pos_tag_index[next_tag])

    for column in range(len(words_list)-2,-1,-1):
      next_word = words_list[column+1]
      max_prev_tag = start_index
      store_max_prob = max_prob
      diff = 1
      for i in range(len(pos_tag_count)):
        prev_tag = i
        cur_prob = dp[i][column]

        if cur_prob != 0:
          transmission_attribute = (pos_tag_index[prev_tag] , pos_tag_index[next_tag])
          transmission_probability = 0.0000000000001
          if transmission_attribute in transmission:
            transmission_probability = transmission[transmission_attribute]


          emission_attribute = (pos_tag_index[next_tag] , next_word)
          emission_probabiity = 0.0000000000001
          if emission_attribute in emission:
            emission_probabiity = emission[emission_attribute]

          if diff > abs(cur_prob*(transmission_probability*emission_probabiity) - max_prob):
            diff = abs(cur_prob*(transmission_probability*emission_probabiity) - max_prob)
            max_prev_tag = prev_tag
            store_max_prob = cur_prob

      next_tag = max_prev_tag
      max_prob = store_max_prob

      predict_tag_array.append(pos_tag_index[max_prev_tag])

    predict_tag_array.reverse()

    for k in range(len(predict_tag_array)):
      output.write(str(k+1)+"\t"+str(words_list[k])+"\t"+str(predict_tag_array[k])+"\n")
    
    words_list = []
