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
	  	#Note: Filenames are not in alphabetical order
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
	def __init__(self, train_filenames, train_labels, test_filenames, train_size = 42000):
		self.train_filenames = train_filenames
		self.train_labels = train_labels
		self.test_filenames = test_filenames
		self.train_size = train_size
		self.val_size = len(filenames) - train_size

	def __parse_function(filenames, labels = None):
	    img_string = tf.read_file(filenames)
	    img_decoded = tf.image.decode_png(img_string, channels=3)
	    
	    #Image augmentation
	    float_image = tf.image.per_image_standardization(img_decoded)
	    
	    img_decoded = tf.reshape(float_image , [-1])  #flatten
	    
	    if labels == None: return img_decoded  #in case of test images
	    else: return img_decoded, labels

	def initTrainIterator(self, batch_size):
		#Creates a initializable object
		train = tf.data.Dataset.from_tensor_slices((self.train_filenames[:self.train_size], self.train_labels[:self.train_size]))
		train = train.map(self.__parse_function, num_parallel_calls = 3)
		train = train.batch(batch_size)
		train = train.shuffle(buffer_size = 1000)    #Keep lower number so that buffer filling happens faster
		self.train_iterator = train.make_initializable_iterator()

	def initValIterator(self):
		#Creates an one shot iterator
		val = tf.data.Dataset.from_tensor_slices((self.train_filenames[self.train_size:], self.train_labels[self.train_size:]))
		val = val.map(self.__parse_function, num_parallel_calls = 3)
		val = val.batch(self.val_size)
		self.val_iterator = val.make_initializable_iterator()

	def initTestIterator(self):
		data_test = tf.data.Dataset.from_tensor_slices(self.test_filenames)
		data_test = data_test.map(_parse_function)
		data_test = data_test.batch(len(self.test_filenames))
		test_iterator = data_test.make_initializable_iterator()
		
	def getTrainBatch(self, batch_size):
		return self.train_iterator.get_next()  

	def getValidationData(self):
		return self.val_iterator.get_next()

	def getTestBatch(self):
		return self.test_iterator.get_next()




if __name__ == '__main__':
	cifar = Cifar10('data/train', 'data/labels.txt', 'data/test','data/labelsTest.csv')
	filenames_train, labels_train = cifar.getTrainData()

	dataHandler = TFDataHandler(filenames_train, labels_train)









