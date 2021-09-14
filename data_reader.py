import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from error_messages import *
import warnings 


class Data:

	def __init__(self, auto_load = True) -> None:
		if auto_load:
			self.load_data()


	def load_data(self, 
				filename = 'modcloth_data.txt', 
				fill_na = False, 
				fill_symbol = None, 
				drop_na = True, 
				ignore_na = False, 
				column_drop_rate = .5,
				drop_cols = []) -> pd.DataFrame():
		# column_drop_rate = max ratio of NaN/Whole data in column needed to drop said column, between 0-1

		if (fill_na or drop_na or ignore_na) is not True:
			raise ArgumentError(f"At least one NaN argument must be true:\nfill_na = {fill_na}  drop_na = {drop_na}   ignore_na = {ignore_na}")
		if (fill_na and drop_na):
			raise ArgumentError(f"Both fill_na and drop_na arguments cannot be simultaneously true")

		if ignore_na:
			if drop_na:
				warnings.warn("Overriding drop_na as False, change ignore_na to False to avoid this\n")
			elif fill_na:
				warnings.warn("Overriding fill_na as False, change ignore_na to False to avoid this\n")
			drop_na = False
			fill_na = False

		try:
			self.dataframe = pd.read_csv(filename)
		except FileNotFoundError:
			raise FileNotFoundError(f"Could not locate {filename}")

		self._drop_columns(column_drop_rate, drop_cols)

		if drop_na:
			self.dataframe.dropna(inplace = True)
		elif fill_na:
			self.dataframe.fillna(fill_symbol, inplace = True)

		self.dataframe.reset_index(inplace = True)
		self.dataframe.drop('index', axis = 1, inplace = True)
		return self.dataframe


	def _drop_columns(self, rate, drop_cols):
		max_non_nans = 0
		for column in drop_cols:
			try:
				self.dataframe.drop(column, axis = 1, inplace = True)
			except KeyError:
				print((f"Couldn't find '{column}' column, continuing\n"))
				
		for column in self.dataframe.columns.values:
			pass
			if len(self.dataframe) - self.dataframe[column].isna().sum() > max_non_nans:
				max_non_nans = len(self.dataframe) - self.dataframe[column].isna().sum()
		for column in self.dataframe.columns.values:
			if self.dataframe[column].isna().sum() / max_non_nans > rate:
				self.dataframe.drop(column, axis = 1, inplace = True)


class EDA:

	def __init__(self, data, style = "dark_background"):
		plt.style.use(style)
		self.data = data


	def ratings_counts(self, show_total = True, show_avg = True):
		counts = self.data.rating.value_counts()
		plt.bar(counts.index.values, counts.values)
		title = 'Ratings'
		if show_total:
			title += f'\nTotal: {sum(counts.values)}'
		if show_avg:
			title += f'\nAverage: {round(np.mean(self.data.rating.values), 2)}'
		plt.title(title)
		plt.ylabel("Count")
		plt.xlabel("Rating")
		plt.tight_layout()
		plt.show()


	def _plotting_helper(self, show, grouping_column, title_text, xlabel, mean = True, reverse = False, sort = True):
		if reverse:
			item_ids = self.data[grouping_column].value_counts().index.values[::-1][:show]
		else:
			item_ids = self.data[grouping_column].value_counts().index.values[:show]
		x = []
		y = []

		items_group = self.data.groupby(grouping_column)
		for _id, table in items_group:
			if _id in item_ids:
				x.append(str(_id))
				if mean:
					y.append(table.rating.mean())
				else:
					y.append(table.rating.count())

		if sort:
			zipped_lists = zip(x, y)
			sorted_pairs = sorted(zipped_lists, key = lambda x: x[1])
			tuples = zip(*sorted_pairs)
			x, y = [list(tuple) for tuple in tuples]

		title_text += f' ({len(x)})\nAverage - {round(np.mean(y), 2)}'
		if len(x) > 20:
			plt.figure(figsize = (len(x) / 3, 5))
		plt.title(title_text)
		plt.xlabel(xlabel)
		plt.ylabel('Rating')
		plt.bar(x, y)
		plt.xticks(rotation = 45)
		plt.tight_layout()
		plt.show()


	def _attributes_helper(self, grouped_column, scaled, title_text):
		feedback_dict = {}
		item_ids = list(self.data['fit'].value_counts().index.values)
		items_group = self.data.groupby(grouped_column)

		for _id, table in items_group:
			fit_counts = table['fit'].value_counts()
			for count in fit_counts:
				if _id in feedback_dict:
					feedback_dict[_id].append(count)
				else:
					feedback_dict[_id] = [count]

		if scaled:
			for k in feedback_dict:
				feedback_dict[k] = [i/max(feedback_dict[k]) for i in feedback_dict[k]]


		x = np.arange(len(item_ids))
		width = .35
		item_ids.insert(0, item_ids[0])

		fig, ax = plt.subplots() # had to instantiate plot this way to be able to edit xticks w strings
		ax.bar(x - width/2, list(feedback_dict.values())[0], width, label = list(feedback_dict.keys())[0])
		ax.bar(x + width/2, list(feedback_dict.values())[1], width, label = list(feedback_dict.keys())[1])
		ax.set_xticklabels(item_ids, rotation = 45)
		ax.set_title(title_text)
		ax.legend()
		ax.set_xlabel("Feedback")
		ax.set_ylabel("Counts")
		plt.tight_layout()
		plt.show()


	def ratings_per_product_most_popular(self, show = 50):
		self._plotting_helper(show, 'item_id', title_text = 'Average Ratings per Product - Most Popular Products', xlabel = 'Product ID', mean = True, reverse = False)

	def ratings_per_product_least_popular(self, show = 50):
		self._plotting_helper(show, 'item_id', title_text = 'Average Ratings per Product - Least Popular Products', xlabel = 'Product ID', mean = True, reverse = True)

	def ratings_count_per_product_most_popular(self, show = 50):
		self._plotting_helper(show, 'item_id', title_text = 'Total Ratings per Product - Most Popular Products', xlabel = 'Product ID',  mean = False, reverse = False)

	def ratings_count_per_product_least_popular(self, show = 50):
		self._plotting_helper(show, 'item_id', title_text = 'Total Ratings per Product - Least Popular Products', xlabel = 'Product ID', mean = False, reverse = True)

	def ratings_per_category(self, show = 4):
		self._plotting_helper(show, 'category', title_text = 'Average Ratings per Category', xlabel = 'Category', mean = True, reverse = False)

	def rating_count_per_category(self, show = 4):
		self._plotting_helper(show, 'category', title_text = 'Total Ratings per Product', xlabel = 'Category', mean = False, reverse = False)

	def rating_per_year(self, show=10):
		self._plotting_helper(show, 'year', title_text = 'Average Rating per Year', xlabel = 'Year', mean = True, sort = False)

	def fit_feedback(self, show = 5):
		self._plotting_helper(show, 'fit', title_text = 'Total Fit Feedback', xlabel = 'Feedback', mean = False)

	def user_attributes(self, show = 2):
		self._plotting_helper(show, 'user_attr', title_text = 'Count of User Attributes', xlabel = 'Attributes', mean = False)

	def model_attributes(self, show = 2):
		self._plotting_helper(show, 'model_attr', title_text = 'Count of Model Attributes', xlabel = 'Attributes', mean = False)

	def fit_feedback_vs_model_attribute(self, scaled = False):
		self._attributes_helper("model_attr", scaled, title_text = 'Fit Feedback vs Model Attributes')

	def fit_feedback_vs_user_attribute(self, scaled = False):
		self._attributes_helper("user_attr", scaled, title_text = 'Fit Feedback vs User Attributes')



if __name__ == '__main__':
	d = Data(auto_load = False)
	df = d.load_data(drop_cols = ['timestamp'])
	
	eda = EDA(df)
	eda.fit_feedback_vs_model_attribute()
	# eda.ratings_count_per_product_most_popular()

'''
 0   item_id     59387 non-null  int64  
 1   user_id     59387 non-null  object 
 2   rating      59387 non-null  int64  
 3   size        59387 non-null  float64
 4   fit         59387 non-null  object 
 5   user_attr   59387 non-null  object 
 6   model_attr  59387 non-null  object 
 7   category    59387 non-null  object 
 8   year        59387 non-null  int64  
 9   split       59387 non-null  int64  
'''