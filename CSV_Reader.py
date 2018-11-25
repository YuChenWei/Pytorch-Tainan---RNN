import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing as skpp
from sklearn.model_selection import train_test_split

class CSVReader:
    def __init__(self):
        self.PassengerList = []
        self.Dataset = []
        self.norDataset = []
        self.Train = []
        self.TrainFeature = []
        self.TrainTarget = []
        self.Test = []
        self.TestFeature = []
        self.TestTarget = []
        self.DataNum = 0
        self.Max = 0
        self.Min = 0
        self.readCSVFile()
        self.covertDataType()
        self.normalizeData()
        self.datasetSplit()
        
    def readCSVFile(self, path = './international-airline-passengers.csv'):
        with open(path, newline = '' ) as csvfile:
            Datas = csv.DictReader(csvfile)
            for row in Datas:
                self.PassengerList.append(row['International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60'])
                
    def covertDataType(self):
        self.DataNum = len(self.PassengerList)-1
        for i in range(self.DataNum):
            self.Dataset.append(float(self.PassengerList[i]))
            
        self.Dataset = np.asarray(self.Dataset)
        #Converting data to float32 for computing with tensorflow
        self.Dataset.astype(np.float32)
        
    # Converting data to normalized data set    
    def normalizeData(self):
        self.scaler = skpp.MinMaxScaler(feature_range = (0, 1))
        self.norDataset = self.scaler.fit_transform(self.Dataset.reshape(-1,1))
        
    # Creating training and testing set    
    def datasetSplit(self):
        # There doesn't need to shuffle the time sequence data.
        self.Train , self.Test = train_test_split(self.norDataset, train_size=0.67, shuffle = False)
        print(len(self.Train))
        self.TrainFeature = self.Train[:len(self.Train)-1]
        print(len(self.TrainFeature))
        self.TrainTarget = self.Train[1:len(self.Train)]
        self.TestFeature = self.Test[:len(self.Test)-1]
        self.TestTarget = self.Test[1:len(self.Test)]


    
