# BERT模型从训练到部署全流程

Tag: BERT 训练 部署
## 缘起
在群里看到许多朋友在使用BERT模型，网上多数文章只提到了模型的训练方法，后面的生产部署及调用并没有说明。
这段时间使用BERT模型完成了从数据准备到生产部署的全流程，在这里整理出来，方便大家参考。

在下面我将以一个“句子相似度”为例子，简要说明从训练到部署的全部流程。最终完成后可以使用一个网页进行交互，实时地对输入的句子对进行相似判断。

## 基本架构

基本架构为：

```mermaid
graph LR
A(BERT模型服务端) --> B(API服务端)
B-->A
B --> C(应用端)
C-->B
```

```
+-------------------+
|   应用端(HTML)    | 
+-------------------+
         ^^
         ||
         VV
+-------------------+
|     API服务端     | 
+-------------------+
         ^^
         ||
         VV
+-------------------+
|  BERT模型服务端   | 
+-------------------+

```


架构说明：
**BERT模型服务端**
	加载模型，进行实时预测的服务；
    使用的是 BERT-BiLSTM-CRF-NER 

**API服务端** 
	调用实时预测服务，为应用提供API接口的服务，用flask编写； 

**应用端**
	最终的应用端；
	我这里使用一个HTML网页来实现；



附件：
本例中训练完成的模型文件.ckpt格式及.pb格式文件，由于比较大，已放到网盘提供下载：
```
链接：https://pan.baidu.com/s/1DgVjRK7zicbTlAAkFp7nWw 
提取码：8iaw 
```
如果你想跳过前面模型的训练过程，可以直接使用训练好的模型，来完成后面的部署。


## 关键节点
主要包括以下关键节点：
* 数据准备
* 模型训练
* 模型格式转化
* 服务端部署与启动
* API服务编写与部署
* 客户端(网页端的编写与部署）


## 数据准备
这里用的数据的数据是从公开数据集下载的，数据比较简单,用来比较text_a text_b 是否相似 0 代表不相似 1代表相似
实验中以$符号分割，以下是以制表符为分割
数据格式如下：

```
text_a	text_b	label
开初婚未育证明怎么弄？	初婚未育情况证明怎么开？	1
谁知道她是网络美女吗？	爱情这杯酒谁喝都会醉是什么歌	0
人和畜生的区别是什么？	人与畜生的区别是什么！	1
请帮我开通一下GPRS套餐	能帮我查询一下GPRS流量还有多少吗	0
这种图片是用什么软件制作的？	这种图片制作是用什么软件呢？	1
这腰带是什么牌子	护腰带什么牌子好	0
什么牌子的空调最好！	什么牌子的空调扇最好	0
校花的贴身高手打给什么时候完结	校花的贴身高手什么时候可以写完	1
移动手机可以改成电信手机吗	能把移动手机改成电信手机吗	1
手机微信内容可以同步到电脑上吗	电脑微信和手机微信可以同步吗	1
哭求《魔术脑》中文版电子书谢谢你们了	中文版《魔术脑》的电子书	1
马上情人节了，大家都怎么安排情人节的行程啊	你将如何安排你的情人节呢？	1
求桃子老师和他的四个学生漫画	跪求桃子老师和四个学生漫画	1
抓耙仔是什么意思	呷仔是什么意思	0
这个表情叫什么	这个猫的表情叫什么	0
介绍几本好看的都市异能小说，要完结的！	求一本好看点的都市异能小说，要完结的	1
一只蜜蜂落在日历上（打一成语）	一只蜜蜂停在日历上（猜一成语）	1
一盒香烟不拆开能存放多久？	一条没拆封的香烟能存放多久。	1
什么是智能手环	智能手环有什么用	0
您好.麻烦您截图全屏辛苦您了.	麻烦您截图大一点辛苦您了.最好可以全屏.	1
苏州达方电子有限公司联系方式	苏州达方电子操作工干什么	0
蛋黄吃多了有什么坏处	吃鸡蛋白过多有什么坏处	0
从龙岗到深圳华夏艺术中心怎么坐车？	从廊坊市里到天津市华夏未来少儿艺术中心怎么走？	0
西安下雪了？是不是很冷啊?	西安的天气怎么样啊？还在下雪吗？	0
第一次去见女朋友父母该如何表现？	第一次去见家长该怎么做	0
猪的护心肉怎么切	猪的护心肉怎么吃	0
显卡驱动安装不了，为什么？	显卡驱动安装不了怎么回事	1
三星手机屏幕是不是最好的？	三星手机的屏幕是不是都很好	0

```

数据按6:2:2的比例拆分成train.csv,test.csv ,dev.csv三个数据文件

## 模型训练
训练模型就直接使用BERT的分类方法，把原来的`run_classifier.py` 。关于训练的代码网上很多，就不展开说明了，主要有以下方法：

```python
#-----------------------------------------
#句子相似度数据处理 liubaobin 2020/8/1
#labels: 0不相似 1相似
class MyProcessor(DataProcessor):
  """Processor for the test data set."""

  def get_train_examples(self, data_dir):
    file_path = os.path.join(data_dir, 'train.csv')
    with open(file_path, 'r', encoding="utf-8") as f:
      reader = f.readlines()
    examples = []
    for index, line in enumerate(reader):
      guid = 'train-%d' % index
      split_line = line.strip().split("$")
      #print(split_line)
      text_a = tokenization.convert_to_unicode(split_line[0])
      text_b = tokenization.convert_to_unicode(split_line[1])
      label = split_line[2]
      examples.append(InputExample(guid=guid, text_a=text_a,
                                       text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    file_path = os.path.join(data_dir, 'val.csv')
    with open(file_path, 'r', encoding="utf-8") as f:
      reader = f.readlines()
    examples = []
    for index, line in enumerate(reader):
       guid = 'val-%d' % index
       split_line = line.strip().split("$")
       text_a = tokenization.convert_to_unicode(split_line[0])
       text_b = tokenization.convert_to_unicode(split_line[1])
       label = split_line[2]
       examples.append(InputExample(guid=guid, text_a=text_a,
                                    text_b=text_b, label=label))
    return examples

  def get_test_examples(self, data_dir):
    """See base class."""
    file_path = os.path.join(data_dir, 'test.csv')
    with open(file_path, 'r', encoding="utf-8") as f:
        reader = f.readlines()
    examples = []
        for index, line in enumerate(reader):
      guid = 'test-%d' % index
      split_line = line.strip().split("$")
      text_a = tokenization.convert_to_unicode(split_line[0])
      text_b = tokenization.convert_to_unicode(split_line[1])
      label = '0'
      examples.append(InputExample(guid=guid, text_a=text_a,
                                       text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
       # Only the test set has a header
       if set_type == "test" and i == 0:
         continue
       guid = "%s-%s" % (set_type, i)
       if set_type == "test":
         text_a = tokenization.convert_to_unicode(line[1])
         label = "0"
       else:
         text_a = tokenization.convert_to_unicode(line[1])
         label = tokenization.convert_to_unicode(line[2])
       examples.append(
         InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

#-----------------------------------------
```
然后添加一个方法：

```python
 processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      'mycorpus': MyProcessor,
  }

```

**特别说明**，这里有一点要注意，在后期部署的时候，需要一个label2id的字典，所以要在训练的时候就保存起来，在`convert_single_example`这个方法里增加一段：

```python
  #--- save label2id.pkl ---
  #在这里输出label2id.pkl , 
  output_label2id_file = os.path.join(FLAGS.output_dir, "label2id.pkl")
  if not os.path.exists(output_label2id_file):
    with open(output_label2id_file,'wb') as w:
      pickle.dump(label_map,w)

  #--- Add end ---
```

这样训练后就会生成这个文件了。

使用以下命令训练模型，目录参数请根据各自的情况修改：

```shell
cd /mnt/sda1/transdat/bert-demo/bert/
export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_L-12_H-768_A-12
export GLUE_DIR=/mnt/sda1/transdat/bert-demo/bert/data
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=mobile_0

sudo python run_mobile.py \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```
由于相关参数不会轻易改动这边直接把参数写到文件里面
只需要执行 python run_classifier.py --do_train=true --do_eval=true 
由于数据比较小，训练是比较快的，训练完成后，可以在输出目录得到模型文件，这里的模型文件格式是.ckpt的。
训练结果：
```
eval_accuracy = 0.861643
eval_f1 = 0.9536328
eval_loss = 0.56324786
eval_precision = 0.9491279
eval_recall = 0.9581805
global_step = 759
loss = 0.5615213

```

可以使用以下语句来进行预测：

```shell
python run_classifier.py --do_predict=true

```


## 模型格式转化

到这里我们已经训练得到了模型，但这个模型是.ckpt的文件格式,文件比较大，并且有三个文件：

```
-rw-r--r-- 1 root root 1227239468 Apr 15 17:46 model.ckpt-759.data-00000-of-00001
-rw-r--r-- 1 root root      22717 Apr 15 17:46 model.ckpt-759.index
-rw-r--r-- 1 root root    3948381 Apr 15 17:46 model.ckpt-759.meta
```

可以看到，模板文件非常大，大约有1.17G。
后面使用的模型服务端，使用的是.pb格式的模型文件，所以需要把生成的ckpt格式模型文件转换成.pb格式的模型文件。
我这里提供了一个转换工具:`freeze_graph.py`，使用如下：
注意,需要把注释的  segment_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'segment_ids')参数加上
只是简单得分类模型可以不用加，由于是判断两个句子是否相似，判断句子之间上下文关系，必须要加

```shell
usage: freeze_graph.py [-h] -bert_model_dir BERT_MODEL_DIR -model_dir
                       MODEL_DIR [-model_pb_dir MODEL_PB_DIR]
                       [-max_seq_len MAX_SEQ_LEN] [-num_labels NUM_LABELS]
                       [-verbose]
```

这里要注意的参数是：

* `model_dir` 就是训练好的.ckpt文件所在的目录
* `max_seq_len` 要与原来一致；
* `num_labels` 是分类标签的个数,本例中是2个


```shell
python freeze_graph.py \
    -bert_model_dir $BERT_BASE_DIR \
    -model_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -max_seq_len 128 \
    -num_labels 3

```

执行成功后可以看到在`model_dir`目录会生成一个`classification_model.pb` 文件。
转为.pb格式的模型文件，同时也可以缩小模型文件的大小,可以看到转化后的模型文件大约是390M。

```
-rw-rw-r-- 1 hexi hexi 409326375 Apr 15 17:58 classification_model.pb
```

## 服务端部署与启动

现在可以安装服务端了，使用的是 bert-base, 来自于项目`BERT-BiLSTM-CRF-NER`, 服务端只是该项目中的一个部分。
项目地址：[https://github.com/macanv/BERT-BiLSTM-CRF-NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER) ，感谢Macanv同学提供这么好的项目。

这里要说明一下，我们经常会看到bert-as-service 这个项目的介绍，它只能加载BERT的预训练模型，输出文本向量化的结果。
而如果要加载fine-turing后的模型，就要用到 bert-base 了，详请请见：
[基于BERT预训练的中文命名实体识别TensorFlow实现](https://blog.csdn.net/macanv/article/details/85684284)


下载代码并安装 ：
```shell
pip install bert-base==0.0.7 -i https://pypi.python.org/simple
```
或者 

```shell
git clone https://github.com/macanv/BERT-BiLSTM-CRF-NER
cd BERT-BiLSTM-CRF-NER/
python3 setup.py install
```


使用 bert-base 有三种运行模式，分别支持三种模型，使用参数`-mode` 来指定：
+ NER      序列标注类型，比如命名实体识别；
+ CLASS    分类模型，就是本文中使用的模型
+ BERT     这个就是跟bert-as-service 一样的模式了

之所以要分成不同的运行模式，是因为不同模型对输入内容的预处理是不同的，命名实体识别NER是要进行序列标注；
而分类模型只要返回label就可以了。


安装完后运行服务，同时指定监听 HTTP 8091端口，并使用GPU 1来跑；

```shell
cd /mnt/sda1/transdat/bert-demo/bert/bert_svr

export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=mobile_0

bert-base-serving-start \
    -model_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -bert_model_dir $BERT_BASE_DIR \
    -model_pb_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -mode CLASS \
    -max_seq_len 128 \
    -http_port 8091 \
    -port 5575 \
    -port_out 5576 \
    -device_map 1 
```
**注意**：port 和 port_out 这两个参数是API调用的端口号，
默认是5555和5556,如果你准备部署多个模型服务实例，那一定要指定自己的端口号，避免冲突。
我这里是改为： 5575 和 5576

如果报错没运行起来，可能是有些模块没装上,都是 bert_base/server/http.py里引用的，装上就好了：

```
sudo pip install flask 
sudo pip install flask_compress
sudo pip install flask_cors
sudo pip install flask_json
```


运行服务后会自动生成很多临时的目录和文件，为了方便管理与启动，可建立一个工作目录，并把启动命令写成一个shell脚本。
这里创建的是`mobile_svr\bertsvr.sh` ，这样可以比较方便地设置服务器启动时自动启动服务，另外增加了每次启动时自动清除临时文件

代码如下：

```shell
#!/bin/bash
#chkconfig: 2345 80 90
#description: 启动BERT分类模型 

echo '正在启动 BERT mobile svr...'
cd /mnt/sda1/transdat/bert-demo/bert/mobile_svr
sudo rm -rf tmp*

export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=mobile_0

bert-base-serving-start \
    -model_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -bert_model_dir $BERT_BASE_DIR \
    -model_pb_dir $TRAINED_CLASSIFIER/$EXP_NAME \
    -mode CLASS \
    -max_seq_len 128 \
    -http_port 8091 \
    -port 5575 \
    -port_out 5576 \
    -device_map 1 

```


CPU得计算效率大概计算了一下，由于公司资源有限，使用的是CPU部署，预测速度为0.25s，训练用得GPU，20W数据大概四五个小时训练完。




## 参考资料:
+ [https://github.com/google-research/bert](https://github.com/google-research/bert)
      
+ [https://github.com/hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service)

+ [https://github.com/macanv/BERT-BiLSTM-CRF-NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER)



