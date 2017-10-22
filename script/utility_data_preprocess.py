# python 3 



import numpy as np
import pandas as pd


def load_data_():
	data_path = '../data/'
	song_extra_info =  pd.read_csv(data_path + 'song_extra_info.csv')
	print (song_extra_info.head(10))
	return  song_extra_info


def load_data():
	data_path = '../data/'
	train = pd.read_csv(data_path + 'train.csv')
	test = pd.read_csv(data_path + 'test.csv')
	songs = pd.read_csv(data_path + 'songs.csv')
	members = pd.read_csv(data_path + 'members.csv')
	song_extra_info =  pd.read_csv(data_path + 'song_extra_info.csv')
	return train, test, songs, members, song_extra_info


def get_time_feature(df):
	df_ = df.copy()
	df_['expiration_year'] = df_['expiration_date'].apply(lambda x : int(str(x)[0:4]))
	df_['expiration_month'] = df_['expiration_date'].apply(lambda x : int(str(x)[4:6]))
	df_['expiration_day'] = df_['expiration_date'].apply(lambda x : int(str(x)[6:8]))
	df_['registration_init_year'] = df_['registration_init_time'].apply(lambda x : int(str(x)[0:4]))
	df_['registration_initn_month'] = df_['registration_init_time'].apply(lambda x : int(str(x)[4:6]))
	df_['registration_initn_day'] = df_['registration_init_time'].apply(lambda x : int(str(x)[6:8]))
	return df_ 














