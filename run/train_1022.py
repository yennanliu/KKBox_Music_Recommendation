# python 3 


#import numpy as np
#import pandas as pd

#from script.utility_data_preprocess import *


def prepare():
	train, test, songs, members, song_extra_info = load_data()
	members_ = get_time_feature(members)


def test():
	data_path = '~/KKBox_Music_Recommendation/data/'
	sample_submission =  pd.read_csv(data_path + 'sample_submission.csv',nrows=10)
	print (sample_submission.head(10))
	return  sample_submission

def test_():
	print ('this is build test')

if __name__ == '__main__':
	#train, test, songs, members, song_extra_info = load_data()
	#load_data_test()
	test_()

