import os
import csv



input_file_name = "/home/Sentence_Similarity/bert/output/test_results.tsv"
input_file_name1 = "/home/Sentence_Similarity/bert/bert_data/test.csv"
print("------bert_model result------!")
like_1=[]
like_test=[]
if os.path.exists(input_file_name):
    data = []; data1 = [] ; i = 0
    stringData = {}

    with open(input_file_name) as csv_file:
        csvreader = csv.reader(csv_file,delimiter="\t")
        for row in csvreader :
            data.append(float(row[1]))
            if  round(float(row[1]),0) == 1.0:
                like_test.append(1.0)
    with open(input_file_name1) as csv_file:
        csvreader = csv.reader(csv_file,delimiter="$")
        for row in csvreader :
            data1.append(float(row[2]))
            if float(row[2]) == 1.0:
                like_1.append(1.0)
    total = len(data)+0.0
    success=0.0
    like_total=len(like_1)+0.0
    like_test_total=len(like_test)+0.0
    print('测试集中语义相似的个数:',like_total)
    for idx in range(0,len(data)-1):
        x = round(data[idx],0)
        if x == data1[idx] and x==1.0:
            success = success+1
    print('其中预测正确相似的个数:',success)
    print('所有测试为相似的个数:',like_test_total)
    print('准确率:',success/like_test_total)
    print('召回率:',success/like_total)
    
   # print("success rate = success/total * 100% = " + str(success/total * 100) + "%")

