## Stephen Shepherd
## 2022-01-21

## module for wrapping data preparation and ml algorithm execution

## figure size for plotting
figsize = (4.5,3)

## imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import time

## functions
def finalize_data(df, target_col_name, test_size=.2, random_state=7, map_binary=False, categorical_features=None, name=None, balance=False, verbose=True):

	## columns to lowercase
	df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)

	## handle nulls
	# categorical attributes
	if categorical_features is not None:
		if target_col_name in categorical_features:
			categorical_features.remove(target_col_name)
		for cf in categorical_features:
			df[cf] = df[cf].fillna('null_value') ## treat null as their own category

	# continuous attributes
	continuous_features = [c for c in df.columns if c != target_col_name]
	if categorical_features is not None:
		continuous_features = [c for c in df.columns if c not in categorical_features + [target_col_name]]
	for c in continuous_features:
		#print('on:', c)
		df[c] = df[c].fillna(np.mean(df[c])) ## fill nulls with means

	## shuffle data
	#df = df.sample(frac=1, random_state=random_state)

	## train/test split
	X = df[[c for c in df.columns if c != target_col_name]]
	y = df[target_col_name]

	if map_binary:
		y = y.map({0:-1,1:1})

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

	if balance:
		## down-sample the negative class
		negative_labels_sample_idx = y_train[y_train != 1].sample(n=y_train[y_train == 1].shape[0], random_state=random_state).index
		positive_labels_idx = y_train[y_train == 1].index
		X_train = pd.concat([X_train.loc[negative_labels_sample_idx], X_train.loc[positive_labels_idx]])
		X_train = X_train.sample(frac=1, random_state=random_state) ## shuffle
		y_train = pd.concat([y_train.loc[negative_labels_sample_idx], y_train.loc[positive_labels_idx]])
		y_train = y_train.loc[X_train.index] ## shuffle

	#print('categorical_features:', categorical_features)

	## encoding of categorical features (defaulting to one-hot as it'll work with most algorithms)
	if categorical_features is not None:
		encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', drop='if_binary')
		encoder.fit(X_train[categorical_features]) ## only allowed to peek at train set
		feat_names_out = encoder.get_feature_names_out(categorical_features)
		encoded_train = encoder.transform(X_train[categorical_features]).toarray()
		encoded_test  = encoder.transform(X_test[categorical_features]).toarray()
		X_train = pd.DataFrame(encoded_train, index=X_train.index, columns=feat_names_out).join(X_train[continuous_features])
		X_test  = pd.DataFrame(encoded_test, index=X_test.index,   columns=feat_names_out).join(X_test[continuous_features])

	X_train.rename(columns={c: c.lower().replace('.','') for c in X_train.columns}, inplace=True)
	X_test.rename( columns={c: c.lower().replace('.','') for c in X_test.columns }, inplace=True)

	## standardize/scale data
	scaler = preprocessing.StandardScaler()
	scaler.fit(X_train) ## only allowed to peek at train set
	scaled_train = scaler.transform(X_train)
	scaled_test  = scaler.transform(X_test)
	X_train = pd.DataFrame(scaled_train, index=X_train.index, columns=X_train.columns)
	X_test  = pd.DataFrame(scaled_test, index=X_test.index,  columns=X_test.columns)

	## all data in dataframe
	train_with_labels = pd.DataFrame(y_train).join(X_train)
	train_with_labels['train_set'] = 'train'

	test_with_labels = pd.DataFrame(y_test).join(X_test)
	test_with_labels['train_set'] = 'test'

	all_data = pd.concat((train_with_labels,test_with_labels))
	all_data = all_data[ ['train_set',target_col_name] + [c for c in all_data.columns if c not in ['train_set',target_col_name]] ]
	all_data.rename(columns={target_col_name:'target'}, inplace=True)

	if verbose:
		print('original shape:', df.shape, 'train size:', X_train.shape, 'test size:', X_test.shape)
		print('training set features:')
		display(all_data.head(1))

	return {
		'name': name,
		'all_data': all_data,
		'X_train': X_train,
		'y_train': y_train,
		'X_test': X_test,
		'y_test': y_test
	}

def get_validation_curve(dataset, estimator, train_sizes=None, logx=None):

	if not train_sizes:
		train_sizes = [1,5,10,25,50,100,150,200,300,400]
		step_size = 100
		if dataset['y_train'].shape[0] > 10000:
			train_sizes += [500,750,1000]
			step_size = 1000
		while dataset['y_train'].shape[0] * .8 > max(train_sizes) + step_size:
			#print(train_sizes)
			train_sizes.append(max(train_sizes) + step_size)
		#print('done')

	train_sizes, train_scores, valid_scores = learning_curve(
		estimator,
		dataset['X_train'],
		dataset['y_train'],
		train_sizes=train_sizes,
		cv=5
	)

	results_df = pd.DataFrame(data={
		'train_size': train_sizes,
		'train_set_accuracy':	 np.mean(train_scores, axis=1),
		'cross_val_accuracy': np.mean(valid_scores, axis=1)
	}).set_index('train_size')

	fig = (
		results_df
		.plot(figsize=figsize, style='.-', ylim=(0,1.05), color=('springgreen','fuchsia'), logx=logx,
			  title=dataset['name'] + ' data, accuracy by # train samples')
	)
	plt.grid(color='lightgray', which='major', axis='y', linestyle='solid')
	plt.hlines(y=dataset['baseline_accuracy'], color='gray', linestyle='--', label='baseline accuracy',
			   xmin=results_df.index.min(), xmax=results_df.index.max())
	plt.xlabel('# observations trained on')
	plt.ylabel('accuracy')
	plt.legend()
	plt.show()


def get_parameter_curve(dataset, estimator, estimator_kwargs, param_info, force_xticks=True, plot_test_set_scores=False,
		plot_time=False, plot_time_same=False, plot_predict_time_same=False, plot_iter=False, ylim=(0,1.05), verbose=False, logx=False, plot_title=None):
	results = []
	best_cv_score = 0
	best_cv_param_val = 0
	
	for pv in param_info['param_values']:

		iter_kwargs = estimator_kwargs
		iter_kwargs[param_info['param_name']] = pv
		clf = clone(estimator)
		clf.set_params(**iter_kwargs)
		cv_score = np.mean( cross_val_score(clone(clf), dataset['X_train'], dataset['y_train'], cv=5, scoring='accuracy') )
		start_time = time.time()
		clf.fit(dataset['X_train'], dataset['y_train'])
		end_time = time.time()
		train_preds = clf.predict(dataset['X_train'])
		predict_start = time.time()
		test_preds  = clf.predict(dataset['X_test'])
		predict_end = time.time()

		train_score = sklearn.metrics.accuracy_score(dataset['y_train'], train_preds)
		test_score  = sklearn.metrics.accuracy_score(dataset['y_test'], test_preds)
		
		if cv_score > best_cv_score:
			best_cv_score = cv_score
			best_cv_param_val = pv

		iter_results = {
			param_info['param_name']: pv,
			'test_set_accuracy': test_score,
			'train_set_accuracy': train_score,
			'cross_val_accuracy': cv_score,
			'training_seconds': end_time - start_time,
			'predict_seconds': predict_end - predict_start
		}

		if plot_iter:
			iter_results['num_iterations'] = clf.n_iter_

		if verbose:
			print(pv, train_score, cv_score)

		results.append(iter_results)

	cols = ['train_set_accuracy','cross_val_accuracy']
	if plot_test_set_scores:
		cols.append('test_set_accuracy')

	print(dataset['name'])
	print('best cross-validation score:', best_cv_score, 'at param value:', best_cv_param_val)
	results_df = pd.DataFrame(results).set_index(param_info['param_name'])[cols]
	#results_df[param_info['param_name']] = results_df[param_info['param_name']].astype(str)
	print('best validation set score  :', max([r['test_set_accuracy'] for r in results]))

	xticks = None
	if force_xticks:
		xticks = results_df.index.tolist()
		if type(xticks[0]) == tuple:
			xticks = [xt[0] for xt in xticks]
		results_df.index = xticks

	if not plot_title:
		plot_title = dataset['name'] + ' data, accuracy by ' + param_info['param_name']

	if plot_time_same:
		results_df = pd.DataFrame(results).set_index(param_info['param_name'])[cols + ['training_seconds']]
		ax = (
			results_df
			[['train_set_accuracy','cross_val_accuracy']]
			.plot(
				figsize=figsize, style='.-', ylim=ylim, color=('lime','fuchsia','lightsteelblue'),
				title=plot_title, logx=logx, ylabel='accuracy', legend=False
			)
		)
		ax.grid(color='lightgray', which='major', axis='y', linestyle='solid')
		ax1 = ax.twinx()
		fig = (
			results_df[['training_seconds']]
			.rename(columns={'training_seconds':'train_seconds (right)'})
			).plot(ax=ax1, color='lightsteelblue', style='.-', ylabel='train_seconds').figure

		## https://stackoverflow.com/questions/33611803/pyplot-single-legend-when-plotting-on-secondary-y-axis
		handles,labels = [],[]
		for ax in fig.axes:
		    for h,l in zip(*ax.get_legend_handles_labels()):
		        handles.append(h)
		        labels.append(l)
		plt.legend(handles,labels, loc="lower right", prop={'size': 9})
		plt.xlabel(param_info['param_name'])
		plt.show()

	elif plot_predict_time_same:
		results_df = pd.DataFrame(results).set_index(param_info['param_name'])[cols + ['predict_seconds']]
		ax = (
			results_df
			[['train_set_accuracy','cross_val_accuracy']]
			.plot(
				figsize=figsize, style='.-', ylim=ylim, color=('lime','fuchsia','lightsteelblue'),
				title=plot_title, logx=logx, ylabel='accuracy', legend=False
			)
		)
		ax.grid(color='lightgray', which='major', axis='y', linestyle='solid')
		ax1 = ax.twinx()
		fig = (
			results_df[['predict_seconds']]
			.rename(columns={'predict_seconds':'predict_seconds (right)'})
			).plot(ax=ax1, color='lightsteelblue', style='.-', ylabel='predict_seconds').figure

		## https://stackoverflow.com/questions/33611803/pyplot-single-legend-when-plotting-on-secondary-y-axis
		handles,labels = [],[]
		for ax in fig.axes:
		    for h,l in zip(*ax.get_legend_handles_labels()):
		        handles.append(h)
		        labels.append(l)
		plt.legend(handles,labels, loc="lower right", prop={'size': 9})
		plt.xlabel(param_info['param_name'])
		plt.show()

	else:
		fig = (
			results_df
			.plot(
				figsize=figsize, style='.-', ylim=ylim, color=('lime','fuchsia'),
				title=plot_title,
				logx=logx
			)
		)
		plt.grid(color='lightgray', which='major', axis='y', linestyle='solid')
		plt.hlines(y=dataset['baseline_accuracy'], color='gray', linestyle='--', label='baseline accuracy',
				   xmin=results_df.index.min(), xmax=results_df.index.max())
		plt.vlines(x=best_cv_param_val, color='lightgray', linestyle=(0,(5,5)), label='best param value',
				   ymin=0, ymax=1)
		plt.xlabel(param_info['param_name'])
		plt.ylabel('accuracy')
		plt.legend()
		plt.show()

	if plot_time:
		secondary_y = None
		cols = ['training_seconds']
		colors = ['cornflowerblue']
		if plot_iter:
			cols.append('num_iterations')
			colors.append('salmon')
			secondary_y = ['iterations']
		
		time_df = pd.DataFrame(results).set_index(param_info['param_name'])[cols]

		if force_xticks:
			xticks = time_df.index.tolist()
			if type(xticks[0]) == tuple:
				xticks = [xt[0] for xt in xticks]
			time_df.index = xticks

		(
		time_df
		.rename(columns={'training_seconds':'train_seconds','num_iterations':'iterations'})
		.plot(figsize=figsize, style='.-', secondary_y=secondary_y, color=colors, ylabel='train_seconds',
			title=dataset['name'] + ' data, train time by ' + param_info['param_name'],
			xticks=xticks, ylim=[0,max(time_df.training_seconds) + .1], xlabel=param_info['param_name'])
		)
		plt.ylabel('iterations')
		plt.xlabel(param_info['param_name'])
		plt.show()


def do_gridsearch(dataset, estimator, estimator_kwargs, param_grid, sample=None):
	print('*** ' + dataset['name'] + ' dataset' + ' ***')

	X_train = dataset['X_train']
	y_train = dataset['y_train']
	if sample:
		X_train = dataset['X_train'].sample(frac=.1, random_state=7)
		y_train = dataset['y_train'].loc[X_train.index]

	print('training shape:', X_train.shape, y_train.shape)
	
	clf = GridSearchCV(estimator, param_grid, n_jobs=-1, verbose=0)
	clf.fit(X_train, y_train)

	train_preds  = clf.predict(dataset['X_train'])
	train_score  = sklearn.metrics.accuracy_score(dataset['y_train'], train_preds)

	test_preds  = clf.predict(dataset['X_test'])
	test_score  = sklearn.metrics.accuracy_score(dataset['y_test'], test_preds)

	print('train accuracy         :', train_score)
	print('best crossval accuracy :', clf.cv_results_["mean_test_score"].max())
	print('validation set accuracy:', test_score)
	print('best params			:\n', clf.best_params_)
	print()

	return clf.best_params_

def get_residuals(dataset, estimator, title_append=''):
	estimator.fit(dataset['X_train'], dataset['y_train'])
	y_preds = estimator.predict(dataset['X_test'])

	resid = pd.DataFrame(data={'y_pred':y_preds, 'y_actual':dataset['y_test']})
	resid['count'] = 1
	resid['correct'] = resid['y_pred'] == resid['y_actual']

	gr = resid.groupby('y_actual')[['correct']].agg(['count','sum'])
	gr['pct_misclassified'] = 1 - gr.correct['sum'] / gr.correct['count']

	gr['pct_misclassified'].plot(kind='bar', figsize=(4,2), color=('mediumblue'),
		title=dataset['name'] + ': validation set error by label ' + title_append,ylim=(0,1.05))
	plt.ylabel('label')
	plt.ylabel('pct_misclassified')
	plt.grid(color='lightgray', which='major', axis='y', linestyle='solid')
	plt.show()

def fit_model(dataset, estimator, estimator_kwargs, run_cv=True):
	clf = clone(estimator)
	clf.set_params(**estimator_kwargs)
	cv_score = None
	if run_cv:
		cv_score = np.mean( cross_val_score(clone(clf), dataset['X_train'], dataset['y_train'], cv=5, scoring='accuracy') )
	start_time = time.time()
	clf.fit(dataset['X_train'],dataset['y_train'])
	end_time = time.time()
	print('train seconds :', end_time - start_time)
	train_preds = clf.predict(dataset['X_train'])
	test_preds =  clf.predict(dataset['X_test'])
	print('crossval score:', cv_score)
	print('train score   :', accuracy_score(dataset['y_train'], train_preds))
	print('test  score   :', accuracy_score(dataset['y_test'], test_preds))

	return clf
