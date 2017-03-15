import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats.mstats import mquantiles
import scipy.stats as ss
import math
import scipy.linalg
import itertools
import copy
import random
import gzip
from sets import Set
#reload(rcca)
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, roc_curve
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
import networkx as nx

def printMatrix(Matrix, ylabel, QuantileValue, LowerUpperLimit, title=''):
    #vmaxLim=mquantiles(Matrix,[0.99])[0]
    Lim=mquantiles(Matrix,[QuantileValue])[0]
    print Matrix.max()
    print np.shape(Matrix)
    print "Limit:", Lim
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.8)
    if LowerUpperLimit == 'lower':
        m = ax.matshow(Matrix, origin="bottom", #norm=colors.LogNorm(),  #norm=colors.SymLogNorm(1),
               cmap="RdYlBu_r", vmin=Lim)
    else:
        m = ax.matshow(Matrix, origin="bottom", #norm=colors.LogNorm(),  #norm=colors.SymLogNorm(1),
               cmap="RdYlBu_r", vmax=Lim) # cmap="RdYlBu_r"


    ax.axhline(-0.5, color="#000000", linewidth=1, linestyle="--")
    ax.axvline(-0.5, color="#000000", linewidth=1, linestyle="--")

    cb = fig.colorbar(m)
    cb.set_label(ylabel)

    ax.set_ylim((-0.5, len(Matrix) - 0.5))
    ax.set_xlim((-0.5, len(Matrix) - 0.5))
    
    plt.title(title)
    plt.show()
    return

def set_diag_to_value(matrix, value):
    np.fill_diagonal(matrix, value)
    return matrix

def binarize(matrix, thres=0):
    matrix2=copy.copy(matrix)
    matrix2[matrix == thres] = 0
    matrix2[matrix > thres] = 1    
    return matrix2

def binarize_w_unlabeled(matrix, thres):
    matrix2=copy.copy(matrix)
    matrix2[matrix == 0] = 0
    matrix2[matrix > thres] = 1 
    matrix2[np.logical_and(matrix>0, matrix<=thres)] = -1
    return matrix2

def demean(d): return d-d.mean(0) 
def zscore(d): return (d-d.mean(0))/d.std(0)

#test proportion is computed as total-train
def shuffle_nodes(vector_1d, trainProportion, valiProportion):
    total_num=len(vector_1d)
    train_num=int(trainProportion*total_num)
    vali_num=int(valiProportion*total_num)
    test_num=total_num-train_num-vali_num
    if test_num<=0:
        print "Nothing in the test set!!!"
    print "Training set: "+str(train_num)
    print "Validation set: "+str(vali_num)
    print "Test set: "+str(test_num)
    #decide the random split of nodes
    nodes=[i for i in range(len(vector_1d))]
    shuffled_nodes=copy.copy(nodes)
    random.shuffle(shuffled_nodes)
    train_nodes=np.array(shuffled_nodes[:train_num])
    vali_nodes=np.array(shuffled_nodes[train_num:train_num+vali_num])
    test_nodes=np.array(shuffled_nodes[train_num+vali_num:])
    return train_nodes, vali_nodes, test_nodes
# shuffle and split nodes 
def train_vali_test(data_3d, labels_1d, indx_1d, trainProportion, valiProportion):
    train_nodes, vali_nodes, test_nodes = shuffle_nodes(labels_1d, trainProportion, valiProportion)
    train_set=data_3d[train_nodes, :, :]
    vali_set=data_3d[vali_nodes, :, :]
    test_set=data_3d[test_nodes, :, :]
    labels_train=labels_1d[train_nodes]
    labels_vali=labels_1d[vali_nodes]
    labels_test=labels_1d[test_nodes]
    indx_train=indx_1d[train_nodes]
    indx_vali=indx_1d[vali_nodes]
    indx_test=indx_1d[test_nodes]
    #train_by_train=PPMatrix[shuffled_nodes[0]][:,shuffled_nodes[0]]
    #print "Train by train:", np.shape(train_by_train), "Test by train:", np.shape(test_by_train), "Test by test:", np.shape(test_by_test), "All by train:", np.shape(all_by_train), "All by all:", np.shape(all_by_all) 
    return train_set, vali_set, test_set, labels_train, labels_vali, labels_test, indx_train, indx_vali, indx_test

def build_distance_for_node(PromoterFile): 
    REFrag_dict={}
    # Assign indices to all promoter HindIII sites.
    for line in open(PromoterFile,'r'):
        words=line.rstrip().split()
        chr = words[0]
        hind3 = (words[1], words[2])
        if chr not in REFrag_dict:
            index=0
            REFrag_dict[chr]={hind3:index}
        else:
            index+=1
        REFrag_dict[chr].update({hind3:index})

    REsiteMids={}
    for chr in REFrag_dict:
        REsiteMids[chr] = np.zeros((len(REFrag_dict[chr]),1))
        for key in REFrag_dict[chr]:
            start = int(key[0])
            end = int(key[1])
            REsiteMids[chr][REFrag_dict[chr][key]] = (start + end)/2
        
    return REsiteMids

# builds adjacency matrix
def BuildMatrix(PromoterFile, InteractionsFile):

    REFrag_dict={}
    # Assign indices to all promoter HindIII sites.
    for line in open(PromoterFile,'r'):
        words=line.rstrip().split()
        chr = words[0]
        hind3 = (words[1], words[2])
        if chr not in REFrag_dict:
            index=0
            REFrag_dict[chr]={hind3:index}
        else:
            index+=1
        REFrag_dict[chr].update({hind3:index})
        
    labels_score={}
    for chr in REFrag_dict:
        uniq=0.0
        non_uniq=0.0
        # Initialize matrix (promoter x promoter)
        labels_score[chr] = np.ones((len(REFrag_dict[chr]), len(REFrag_dict[chr]))) #  number of promoters in chr 
        labels_score[chr] = labels_score[chr]*(-1)
 
    total_lines=0.0
    for line in gzip.open(InteractionsFile,'r'):
        words=line.rstrip().split()
        chr = words[0]
        hind3_1 = (words[1], words[2])
        hind3_2 = (words[4], words[5])
        q_values = float(words[6])

        i=REFrag_dict[chr][hind3_1]
        j=REFrag_dict[chr][hind3_2]
        
        if labels_score[chr][i,j] != -1:
            non_uniq+=1
            # mean
            labels_score[chr][i,j]=labels_score[chr][i,j]/2+q_values/2
            labels_score[chr][j,i]=labels_score[chr][j,i]/2+q_values/2
#             # max
#             labels_score[chr][i,j]=max(labels_score[chr][i,j], q_values)
#             labels_score[chr][j,i]=max(labels_score[chr][j,i], q_values)
        else:
            uniq +=1
            labels_score[chr][i,j]=q_values
            labels_score[chr][j,i]=q_values
            
        total_lines += 1
    
    print "non-unique entries in CaptureC file(bait1-bait2/bait2-bait1): ", non_uniq, " ", non_uniq/total_lines, "%"
    print "unique entries in CaptureC file(bait1-bait2/bait2-bait1): ", uniq, " ", uniq/total_lines, "%"
    return labels_score

# builds feature vector 
def get_features(PromoterFile, FeatureVectorFile, dataName):

    REFrag_dict={}
    # Assign indices to all promoter HindIII sites.
    for line in open(PromoterFile,'r'):
        words=line.rstrip().split()
        chr = words[0]
        hind3 = (words[1], words[2])
        if chr not in REFrag_dict:
            index=0
            REFrag_dict[chr]={hind3:index}
        else:
            index+=1
        REFrag_dict[chr].update({hind3:index})

    # Initialize vector (promoters only)
    vector=np.zeros((len(REFrag_dict),)) #  number of promoters in chr 1

    features={}
    for line in open(FeatureVectorFile,'r'):              
        words=line.rstrip().split()
        chr = words[0]
        hind3 = (words[1], words[2])
        value = words[3]
        if chr not in features:
            # Initialize vector (promoters only)
            features[chr] = np.zeros((len(REFrag_dict[chr]))) #  number of promoters in chr 
            features[chr][REFrag_dict[chr][hind3]] = value
        else:
            features[chr][REFrag_dict[chr][hind3]] = value
            

    # list of non-zero q-values
    nonzero_values=filter(lambda a: a != 0, np.array(features['chr1']).reshape(-1,).tolist()) 

    # Print out average q-values:
    print "Average value with zeros: ", str(np.average(features['chr1']))
    print "Average q-value w/o zeros: ", np.mean(nonzero_values)

    # Print distribution of q-values
    plt.hist(features['chr1'])
    plt.title(str(dataName))
    plt.show()

    return features

def get_data_labels(FeatureVector, labels_score):
    data={}
    labels={}
    indx={}
    for chr in FeatureVector:
        k=0
        data[chr]=np.zeros(((FeatureVector[chr].shape[0]**2-FeatureVector[chr].shape[0])/2,11,2))
        labels[chr]=np.zeros(((FeatureVector[chr].shape[0]**2-FeatureVector[chr].shape[0])/2,1))
        indx[chr]=np.zeros(((FeatureVector[chr].shape[0]**2-FeatureVector[chr].shape[0])/2,3))
        for i in range(FeatureVector[chr].shape[0]):
            for j in range(FeatureVector[chr].shape[0]):
                if i < j:
                    data[chr][k,:,0]=FeatureVector[chr][i,:]
                    data[chr][k,:,1]=FeatureVector[chr][j,:]
                    labels[chr][k]=labels_score[chr][i][j]
                    indx[chr][k,0]=i
                    indx[chr][k,1]=j
                    if chr=='chrX':
                        indx[chr][k,2]=23
                    else:
                        indx[chr][k,2]=chr[3:]
                    k+=1
    return data, labels, indx

def remove_unlabeled(data, labels, indx):
    indx_unlabeled = np.where(labels==-1)[0]
    new_labels=np.delete(labels, indx_unlabeled, 0)
    new_data=np.delete(data, indx_unlabeled, 0)
    new_indx=np.delete(indx, indx_unlabeled, 0)
    return new_data, new_labels, new_indx


def concatenate_chrs(data, labels, indx):
    data_to_concatenate = []
    labels_to_concatenate = []
    indx_to_concatenate = []
    for chr in data:
        data_to_concatenate.append(data[chr])
        labels_to_concatenate.append(labels[chr])
        indx_to_concatenate.append(indx[chr])
    return np.concatenate(data_to_concatenate, axis=0), np.concatenate(labels_to_concatenate, axis=0), np.concatenate(indx_to_concatenate, axis=0)

def get_pairs_distance_matched(X, y, indx, min_dist, max_dist, dist_step, imbalance_ratio):
    
    def subsample_indx(indecies, size, imbalance_ratio):
        indecies_shuffled=copy.copy(indecies)
        np.random.shuffle(indecies_shuffled)
        num_subsampled = size*imbalance_ratio
        if num_subsampled > len(indecies[0]):
            print '    Error: Not enough to subsample'
            exit 
        #print indecies_shuffled[0].shape
        #print indecies_shuffled[0][:num_subsampled].shape
        else:
            return indecies_shuffled[0][:num_subsampled]

    neg_indxs = np.where(y==0)[0]
    pos_indxs = np.where(y==1)[0]
    X_pos=X[pos_indxs]
    X_neg=X[neg_indxs]
    y_pos=y[pos_indxs]
    y_neg=y[neg_indxs]
    indx_pos=indx[pos_indxs]
    indx_neg=indx[neg_indxs]
    
    thres1=min_dist+dist_step
    thres2=min_dist
    
    X_new=np.empty(([0,X.shape[1],X.shape[2]]))
    y_new=np.empty(([0,y.shape[1]]))
    indx_new=np.empty(([0,indx.shape[1]]))
    
    while thres1 <= max_dist:
        print 'distance window: ', '[', thres2, ',', thres1, ']'
        neg_indx_at_dist=np.where((abs(X_neg[:, 10, 0] - X_neg[:, 10, 1]) <= thres1) & (abs(X_neg[:, 10, 0] - X_neg[:, 10, 1]) >= thres2))
        pos_indx_at_dist=np.where((abs(X_pos[:, 10, 0] - X_pos[:, 10, 1]) <= thres1) & (abs(X_pos[:, 10, 0] - X_pos[:, 10, 1]) >= thres2))

        if len(pos_indx_at_dist[0])> len(neg_indx_at_dist[0]):
            #print 'more pos than neg'
            print 'number of pos at distance=:',  len(pos_indx_at_dist[0])
            print 'number of neg at distance=:',  len(neg_indx_at_dist[0])
            indx_subsampled=subsample_indx(pos_indx_at_dist, len(neg_indx_at_dist[0]), imbalance_ratio)
            new_pos_indx_at_dist=indx_subsampled 
            new_neg_indx_at_dist=neg_indx_at_dist[0]

        else:
            #print 'more neg than pos'
            print 'number of pos at distance=',  len(pos_indx_at_dist[0])
            print 'number of neg at distance=',  len(neg_indx_at_dist[0])
            indx_subsampled=subsample_indx(neg_indx_at_dist, len(pos_indx_at_dist[0]), imbalance_ratio)
            new_pos_indx_at_dist=pos_indx_at_dist[0]
            new_neg_indx_at_dist=indx_subsampled
         
        y_pos_at_dist=y_pos[new_pos_indx_at_dist]
        y_neg_at_dist=y_neg[new_neg_indx_at_dist]
        X_pos_at_dist=X_pos[new_pos_indx_at_dist]
        X_neg_at_dist=X_neg[new_neg_indx_at_dist]
        indx_pos_at_dist=indx_pos[new_pos_indx_at_dist]
        indx_neg_at_dist=indx_neg[new_neg_indx_at_dist]

        y_at_dist=np.concatenate((y_pos_at_dist, y_neg_at_dist))
        X_at_dist=np.concatenate((X_pos_at_dist, X_neg_at_dist))
        indx_at_dist=np.concatenate((indx_pos_at_dist, indx_neg_at_dist))
        
        print 'labels at dist: ', y_at_dist.shape
        print 'data at dist: ', X_at_dist.shape 
        print 'indx at dist: ', indx_at_dist.shape

        indx_new=np.concatenate((indx_new, indx_at_dist))
        X_new=np.concatenate((X_new, X_at_dist))
        y_new=np.concatenate((y_new, y_at_dist))
        
        
        #print X_new.shape, X_at_dist.shape
        #print y_new.shape, y_at_dist.shape
        #print indx_new.shape, indx_at_dist.shape
        
        #print "# of neg:", np.where(y_at_dist==0)[0].shape
        #print "# of pos:", np.where(y_at_dist==1)[0].shape
        
        #thres2=thres1+min_dist
        thres2=thres1
        thres1=thres1+dist_step
              
    return X_new, y_new, indx_new

def impose_dist_constrains(data, labels, indx, min_dist, max_dist):
    indx_unlabeled = np.where((abs(data[:, 10, 0] - data[:, 10, 1]) <= max_dist) & (abs(data[:, 10, 0] - data[:, 10, 1]) >= min_dist))
    new_labels=labels[indx_unlabeled]
    new_data=data[indx_unlabeled]
    new_indx=indx[indx_unlabeled]
    return new_data, new_labels, new_indx

def count_nodes_and_contacts(indx_train_distance_matched, indx_valid_distance_matched, indx_test_distance_matched):
    indx=np.concatenate((indx_train_distance_matched, indx_valid_distance_matched, indx_test_distance_matched), axis=0)
    sum=0
    print "Number of distance-matched links in each chr:"
    for i in range(23):
    	x=np.ravel(np.where(indx[:,2]==i+1))
    	print "chr"+str(i+1)+":", x.shape[0]
    	sum=sum+int(x.shape[0])
    print "Total:", sum

    sum=0
    nodes=Set([])
    print "Number of nodes in each chr:"
    for i in range(23):
        x=np.ravel(np.where(indx[:,2]==i+1))
        nodes.clear()
        nodes=Set(indx[np.where(indx[:,2]==i+1)][:,0])
        nodes.update(Set(indx[np.where(indx[:,2]==i+1)][:,1]))
        print "chr"+str(i+1)+":", len(nodes)
        sum=sum+int(len(nodes))
    print "Total:", sum
