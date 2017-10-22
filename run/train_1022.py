# python 3 


import numpy as np
import pandas as pd

from script.utility_data_preprocess import *


def prepare():
	train, test, songs, members, song_extra_info = load_data()
	members_ = get_time_feature(members)


if __name__ == '__main__':
	#train, test, songs, members, song_extra_info = load_data()
	load_data_test()


