import json

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Config(object):
    sequenceLength = 200
    batchSize = 128
    numClasses = 1
    rate = 0.8

config = Config()

# 注：下面两个词典要保证和当前加载的模型对应的词典是一致的
with open("E:/python_workspaces/lstm/data/wordJson/word2idx.json", "r", encoding="utf-8") as f:
    word2idx = json.load(f)

with open("E:/python_workspaces/lstm/data/wordJson/label2idx.json", "r", encoding="utf-8") as f:
    label2idx = json.load(f)
idx2label = {value: key for key, value in label2idx.items()}


graph = tf.Graph()
with graph.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        checkpoint_file = tf.train.latest_checkpoint("../model/Bi-LSTM/model/")
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 获得需要喂给模型的参数，输出的结果依赖的输入值
        inputX = graph.get_operation_by_name("inputX").outputs[0]
        dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]

        # 获得输出的结果
        predictions = graph.get_tensor_by_name("output/predictions:0")

        # for i in range(10):
        x = "this movie is full of references like mad max ii the wild one and many others the ladybug´s face it´s a clear reference or tribute to peter lorre this movie is a masterpiece we´ll talk much more about in the future"
        xIds = [word2idx.get(item, word2idx["UNK"]) for item in x.split(" ")]
        if len(xIds) >= config.sequenceLength:
            xIds = xIds[:config.sequenceLength]
        else:
            xIds = xIds + [word2idx["PAD"]] * (config.sequenceLength - len(xIds))
        pred = sess.run(predictions, feed_dict={inputX: [xIds], dropoutKeepProb: 1.0})[0]
        print(pred)
pred = [idx2label[item] for item in pred]
print(idx2label)
print(pred)