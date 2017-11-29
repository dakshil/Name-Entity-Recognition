#!/usr/bin/env python
#title           :NER.py
#description     :Train and create dmemm model with evaluation.
#author          :Dakshil Shah
#usage           :python NER.py
#notes           : python 2.7, with pytorch 
#==============================================================================

# Import the modules needed to run the script.import subprocess
import argparse
import sys
import gzip
import cPickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
from torch.autograd import Variable
DIM=10

def get_embeddings(idx2Text,vocab_size, dimensions):
    embeds = nn.Embedding(vocab_size, dimensions)  #  words in vocab, 5 dimensional embeddings
    keys = idx2Text.keys()
    id2embedding = {}
    for i in keys:
        id2embedding[i]=embeds(autograd.Variable(torch.LongTensor([i])))
    return id2embedding

def create_word_embeddings(idx2word,dimensions):
    vocab_size = len(idx2word)
    word_embeddings=get_embeddings(idx2word,vocab_size,dimensions)
    torch.save(word_embeddings, 'wordEmbeddings.pt')

def create_label_embeddings(idx2label,dimensions):
    labels_size = len(idx2label)
    label_embeddings=get_embeddings(idx2label,labels_size,dimensions)
    torch.save(label_embeddings, 'labelEmbeddings.pt')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(15, 150) #10 dim of input, 5 for word, 5 for label, 150 size of hidden
        self.fc2 = nn.Linear(150, 127) # hidden layer to 127 output labels

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

def create_nn(batch_size,opt,dimensions, learning_rate, epochs,log_interval,trainTuplesConcat,targetTuplesConcat,word_embeddings,label_embeddings,trainModelFlag,targetTuplesConcatIndex):
    DIM=15
    if trainModelFlag:
        net = Net()
        print(net)
        # create a stochastic gradient descent optimizer
        if(opt=="SGD"):
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        elif(opt=="ADA"):
            optimizer = optim.Adadelta(net.parameters(), lr=1.0, eps=1e-06, weight_decay=0)
        elif(opt=="ADAM"):
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        # create a loss function
        criterion = nn.NLLLoss()
        batch_sizeOrg=batch_size
        # run the main training loop
        for epoch in range(epochs):
            batch_size=batch_sizeOrg
            for i in range(0,len(trainTuplesConcat),batch_size):
                print i
                if(i+batch_size > len(trainTuplesConcat)):
                    batch_size = len(trainTuplesConcat) - i
                optimizer.zero_grad()
                data = autograd.Variable(trainTuplesConcat[i:(i+batch_size)].data.view(batch_size,15),requires_grad=True)
                # target= autograd.Variable(targetTuplesConcat[i:(i+batch_size)].data)
                target= targetTuplesConcatIndex[i:(i+batch_size)]
                # print target
                # target_keys=[]
                # for k in range(batch_size):
                #     target_keys.append(get_key(target[k],label_embeddings))
                target_keys=target
                exp=autograd.Variable(torch.LongTensor(target_keys))
                # print data
                # print exp
                # print net(data)
                # _, predictedTop5_test = net(data).topk(5)
                # _, prediction = net_out.topk(1)
                # print prediction.data.numpy()[0][0], get_key(target,label_embeddings)
                # print predictedTop5_test
                # break
                # loss = criterion(prediction.data.numpy()[0], get_key(target,label_embeddings))
                # loss=criterion(label_embeddings[prediction.data.numpy()[0][0]],target)
                loss= criterion(net(data), exp)
                loss.backward()
                optimizer.step()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, i, len(trainTuplesConcat),
                               100. * i / len(trainTuplesConcat), loss.data[0]))
        torch.save(net.state_dict(),"neural_net_modelADAM4.pt")


def validationNN(dimensionsOrg,idx2word,idx2label,word_embeddings,label_embeddings,valid_lex,valid_y):
    net=Net()
    net.load_state_dict(torch.load("neural_net_modelADAM4.pt"))
    criterion = nn.NLLLoss()
    dimensions=dimensionsOrg*3
    greedy=False
    viterbi=True
    if greedy:
        totalOutput=[]
        for i in range(len(valid_lex)):
            output=[] 
            prev_label=label_embeddings[127]
            wordEPrev=prev_label
            for j in range(len(valid_lex[i])):
                word_embed=torch.cat((word_embeddings[valid_lex[i][j]],wordEPrev,prev_label),1)
                true_label=valid_y[i][j]
                data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                prediction = net(data)
                predictionTop5_logProp, predictedTop5_Keys = prediction.topk(5)
                output.append(predictedTop5_Keys[0,0].data.numpy()[0])
                wordEPrev=word_embeddings[valid_lex[i][j]]
                #####greedy#####
                prev_label=label_embeddings[predictedTop5_Keys[0,0].data.numpy()[0]]
                #################
            predictions_test = map(lambda t: idx2label[t], output)
            totalOutput.append(predictions_test)
        return totalOutput
    elif viterbi:
        totalOutput=[]
        rows=len(label_embeddings)-1
        for i in range(len(valid_lex)):
            # print i
            #FORWARD PASS TO CREATE DP TABLE
            cols=len(valid_lex[i])
            #-1 to account for start label
            viterbiProbTable = numpy.zeros(shape=(rows,cols))
            viterbiBackBackTable = numpy.zeros(shape=(rows,cols))
            for j in range(cols):
                if(j==0):
                    #if first word then prev label is only start label
                    prev_label=(label_embeddings[127])
                    word_embed=torch.cat((word_embeddings[valid_lex[i][j]],prev_label,prev_label),1)
                    data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                    prediction = net(data)
                    # print prediction.data.view(127,1).numpy().shape
                    colProb=prediction.data.view(127).numpy()
                    viterbiProbTable[:,j]=colProb
                    viterbiBackBackTable[:,j]=128
                    
                elif(j!=0):
                    for k in range(rows):
                        prev_label=(label_embeddings[k])
                        word_embed=torch.cat((word_embeddings[valid_lex[i][j]],word_embeddings[valid_lex[i][j-1]],prev_label),1)
                        data = autograd.Variable(word_embed.data.view(1,dimensions),requires_grad=True)
                        prediction = net(data)
                        colProb=prediction.data.view(127).numpy()
                        if k==0:
                            viterbiProbTable[:,j]=colProb+viterbiProbTable[k][j-1]
                            viterbiBackBackTable[:,j]=k
                        else:
                            for x in range(rows):
                                if(viterbiProbTable[x][j]<colProb[x]+viterbiProbTable[k][j-1]):
                                    viterbiProbTable[x][j]=colProb[x]+viterbiProbTable[k][j-1]
                                    viterbiBackBackTable[x][j]=k
            # print viterbiProbTable
            # print viterbiBackBackTable
            #BACKWARD PASS TO CREATE PATH
            output=[]
            for j in range(cols-1,-1,-1):
                if j==cols-1:
                    row_index = viterbiProbTable[:,j].argmax(axis=0)
                    output.append(row_index)
                    prevLabel=viterbiBackBackTable[row_index][j]
                    # print prevLabel
                else:
                    output.append(prevLabel)
                    # print viterbiBackBackTable[int(prevLabel)][j]
                    prevLabel=viterbiBackBackTable[int(prevLabel)][j]
            output.reverse()
            # print output,valid_y[i]
            predictions_test = map(lambda t: idx2label[t], output)
            # print predictions_test
            totalOutput.append(predictions_test)
        return totalOutput

    return []

def trainNeuralNet(trainTuplesConcat,targetTuplesConcat,word_embeddings,label_embeddings,targetTuplesConcatIndex):
    '''
    ADA
    0.001 lr
    800 epochs
    batch 2000
    50 hidden
    neural_net_model.pl


    ADAM
    0.1 lr
    100 epoch
    batch 10000
    150 hidden
    neural_net_modelADAM.pl
    '''
    opt="ADAM"
    batch_size=10000
    dimensions=10
    learning_rate=0.01
    epochs=1000
    log_interval=1000
    trainModelFlag= True
    create_nn(batch_size, opt, dimensions ,learning_rate, epochs,log_interval,trainTuplesConcat,targetTuplesConcat,word_embeddings,label_embeddings,trainModelFlag,targetTuplesConcatIndex)

def startTraining(idx2word,idx2label,word_embeddings,label_embeddings,train_lex,train_y):
    createTotalEmbedding=False
    if(createTotalEmbedding):
        for i in range(len(train_lex)):
            # print train_lex[i], map(lambda t: idx2word[t], train_lex[i])
            # print train_y[i], map(lambda t: idx2label[t], train_y[i])
            # print "\n"
            for j in range(len(train_lex[i])):
                #concat word and label
                if j==0:
                    # print word_embeddings[train_lex[i][j]]
                    # print label_embeddings[127]
                    #concat first word with prev label IE START Label
                    #if first word then put start label as the previous word
                    trainTuple=torch.cat(( word_embeddings[train_lex[i][j]],label_embeddings[127],label_embeddings[127]),1)
                    if i==0:
                        trainTuplesConcat=trainTuple
                        targetTuplesConcat=label_embeddings[train_y[i][j]]
                        targetTuplesConcatIndex=torch.LongTensor([train_y[i][j].tolist()])
                    else: 
                        trainTuplesConcat=torch.cat((trainTuplesConcat,trainTuple),0)
                        targetTuplesConcat=torch.cat((targetTuplesConcat,label_embeddings[train_y[i][j]]),0)
                        targetTuplesConcatIndex=torch.cat((targetTuplesConcatIndex,torch.LongTensor([train_y[i][j].tolist()])),0)
                else:
                    # print word_embeddings[train_lex[i][j]]
                    # print label_embeddings[train_y[i][j-1]]
                    #concat second word onwards with prev label
                    #consider the previous word too
                    trainTuple=torch.cat(( word_embeddings[train_lex[i][j]],word_embeddings[train_lex[i][j-1]],label_embeddings[train_y[i][j-1]]),1)
                    # print trainTuple
                    trainTuplesConcat=torch.cat((trainTuplesConcat,trainTuple),0)
                    targetTuplesConcat=torch.cat((targetTuplesConcat,label_embeddings[train_y[i][j]]),0)
                    targetTuplesConcatIndex=torch.cat((targetTuplesConcatIndex,torch.torch.LongTensor([train_y[i][j].tolist()])),0)
                # print trainTuple
                # print "\n"
        torch.save(trainTuplesConcat, 'trainTuplesEmbeddings.pt')
        torch.save(targetTuplesConcat, 'trainLabelsEmbedding.pt')
        torch.save(targetTuplesConcatIndex, 'trainLabelsIndex.pt')
        # torch.save(targetClassConcat, 'trainLabelsClass.pt')

    trainTuplesConcat = torch.load('trainTuplesEmbeddings.pt') 
    targetTuplesConcat = torch.load('trainLabelsEmbedding.pt')
    targetTuplesConcatIndex=torch.load('trainLabelsIndex.pt')
    # targetClassConcat = torch.load('trainLabelsClass.pt')

    trainNeuralNet(trainTuplesConcat,targetTuplesConcat,word_embeddings,label_embeddings,targetTuplesConcatIndex)
    

def conlleval(p, g, w, filename='tempfile.txt'):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, ww in zip(sl, sp, sw):
            out += ww + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain precision/recall and F1 score '''
    _conlleval = 'conlleval.pl'

    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall    = float(out[8][:-2])
    f1score   = float(out[10])

    return (precision, recall, f1score)

def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", type=str, default="atis.small.pkl.gz", help="The zipped dataset")

    parsed_args = argparser.parse_args(sys.argv[1:])

    filename = parsed_args.data
    f = gzip.open(filename,'rb')
    train_set, valid_set, test_set, dicts = cPickle.load(f)

    train_lex, _, train_y = train_set
    valid_lex, _, valid_y = valid_set
    test_lex,  _,  test_y  = test_set

    idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())
    #add a start label to the labels at indx 127
    idx2label[len(idx2label)]='START'

    # print idx2word
    # print idx2label
    # print "\n"
    '''
    To have a look what the original data look like, comment them before your submission
    '''
    # print test_lex[0], map(lambda t: idx2word[t], test_lex[0])
    # print "\n"
    # print test_y[0], map(lambda t: idx2label[t], test_y[0])
    # print "\n"
    # print train_y[0], map(lambda t: idx2label[t], train_y[0])

    '''
    implement you training loop here
    '''
    #create word embeddings
    embedFlag=False
    dimensionsOrg=5
    if(embedFlag):
        create_word_embeddings(idx2word,dimensionsOrg)
        create_label_embeddings(idx2label,dimensionsOrg)

    word_embeddings = torch.load('wordEmbeddings.pt') 
    label_embeddings = torch.load('labelEmbeddings.pt')

    trainFlag=False
    if(trainFlag):
        startTraining(idx2word,idx2label,word_embeddings,label_embeddings,train_lex,train_y)
    validFlag=False
    if(validFlag):
        # predictions_val=validationNN(dimensionsOrg,idx2word,idx2label,word_embeddings,label_embeddings,valid_lex,valid_y)
        predictions_val=validationNN(dimensionsOrg,idx2word,idx2label,word_embeddings,label_embeddings,valid_lex,valid_y)

        # print predictions_val[0]
        groundtruth_val = [ map(lambda t: idx2label[t], y) for y in valid_y ]
        # print groundtruth_val[0]
        words_val = [ map(lambda t: idx2word[t], w) for w in valid_lex ]
        test_precision, test_recall, test_f1score = conlleval(predictions_val, groundtruth_val, words_val)

        print test_precision, test_recall, test_f1score
    '''
    how to get f1 score using my functions, you can use it in the validation and training as well
    '''
    testFlag=True
    if(testFlag):
        # predictions_val=validationNN(dimensionsOrg,idx2word,idx2label,word_embeddings,label_embeddings,valid_lex,valid_y)
        predictions_val=validationNN(dimensionsOrg,idx2word,idx2label,word_embeddings,label_embeddings,test_lex,test_y)

        # print predictions_val[0]
        groundtruth_val = [ map(lambda t: idx2label[t], y) for y in test_y ]
        # print groundtruth_val[0]
        words_val = [ map(lambda t: idx2word[t], w) for w in test_lex ]
        test_precision, test_recall, test_f1score = conlleval(predictions_val, groundtruth_val, words_val)

        print test_precision, test_recall, test_f1score


if __name__ == '__main__':
    main()
