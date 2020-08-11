import os
import csv



input_file_name = "/home/Sentence_Similarity/bert/output/test_results.tsv"
input_file_name1 = "/home/Sentence_Similarity/bert/bert_data/test.csv"
output_file_name = "/home/Sentence_Similarity/bert/output/test_result_check.txt"
print("start------!")
if os.path.exists(input_file_name):
    print("now check file"+input_file_name)
    data = []; data1 = [] ; i = 0
    stringData = {}

    with open(input_file_name) as csv_file:
        csvreader = csv.reader(csv_file,delimiter="\t")
        for row in csvreader :
            data.append(float(row[1]))
    print("now reading file:"+input_file_name1)
    with open(input_file_name1) as csv_file:
        csvreader = csv.reader(csv_file,delimiter="$")
        for row in csvreader :
            data1.append(float(row[2]))
    total = len(data)+0.0
    success=0.0
    print("read file ok!")
    for idx in range(0,len(data)-1):
        x = round(data[idx],0)
        if x == data1[idx] :
            success = success+1

    print("success rate = success/total * 100% = " + str(success/total * 100) + "%")

