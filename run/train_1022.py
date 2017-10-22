# python 3 


import numpy as np
import pandas as pd

from script.utility_data_preprocess import *


def prepare():
	train, test, songs, members, song_extra_info = load_data()
	members_ = get_time_feature(members)


def test():
	data_path = '~/KKBox_Music_Recommendation/data/'
	song_extra_info =  pd.read_csv(data_path + 'song_extra_info.csv',nrows=10)
	print (song_extra_info.head(10))
	return  song_extra_info

if __name__ == '__main__':
	#train, test, songs, members, song_extra_info = load_data()
	#load_data_test()
	test()

