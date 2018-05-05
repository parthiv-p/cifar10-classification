"""
File to handle all cifar10 data
"""
#TODO: Add support to return test labels
import tensorflow as tf
import glob

class Cifar10:
	def __init__(self, train_path, train_labels_path, test_path, test_labels_path):
		self.data_path_train = train_path
		self.data_path_labels = train_labels_path
		self.data_path_test = test_path
		self.data_path_test_labels = test_labels_path
	  	#get all filenames
	  	#Note that the filenames are not in alphabetical order
		self.filenames = [file for file in glob.glob(self.data_path_train + '*/*')]
		
	def __get_label_mapping(self, label_file):
		"""
		Returns mappings of label to index and index to label
		The input file has list of labels, each on a separate line.
		"""
		with open(label_file, 'r') as f:
			id2label = f.readlines()
			id2label = [l.strip() for l in id2label]
			label2id = {}
			count = 0
			for label in id2label:
				label2id[label] = count
				count += 1
		return id2label, label2id

	def getLabels(self):
		labels = [file.split('_')[-1].split('.')[0] for file in self.filenames] #get label string name eg. frog, ship...
		id2label, label2id = self.__get_label_mapping(self.data_path_labels) 
		labels = [label2id[label] for label in labels if label in label2id] #map label  name to id number eg  ship -> 8
		return labels

	def getTrainData(self):
		labels = self.getLabels()
		return self.filenames, labels

	def getTestData(self):
		filenames_test = [file_t for file_t in glob.glob(self.data_path_test + '*/*')]  
		filenames_test.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  #sort data so that it is not random
		return filenames_test


class TFDataHandler():
	def __init__(self, filenames, labels):
		self.filenames = filenames
		self.labels = labels

if __name__ == '__main__':
	cifar = Cifar10('data/train', 'data/labels.txt', 'data/test','data/labelsTest.csv')
	filenames_train, labels_train = cifar.getTrainData()

	dataHandler = TFDataHandler(filenames_train, labels_train)









