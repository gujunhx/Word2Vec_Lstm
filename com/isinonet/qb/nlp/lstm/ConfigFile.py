
class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001

class ModelConfig(object):
    embeddingSize = 200
    hiddenSize = [256, 256]
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0

class Config(object):
    dataSource = "E:/python_workspaces/examp/data/preProcess/labeledTrain.csv"
    stopWordSource = "E:/python_workspaces/examp/data/english"

    sequenceLength = 200
    batchSize = 128
    numClasses = 1
    rate = 0.8
    model = ModelConfig()
    training = TrainingConfig()