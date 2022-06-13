import argparse
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader
import numpy as np
import matplotlib.pyplot as plt
import pdb # mz
from numpy import zeros

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to input dataset')
ap.add_argument('-n', '--neighbors', required=False, type=int, default=1,
                help='# of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', required=False, type=int, default=-1,
                help='# of jobs for k-NN distance (-1 uses all available cores)')

args = vars(ap.parse_args())

# Get list of image paths
image_paths = list(paths.list_images(args['dataset']))
eval_image_paths = list(paths.list_images('../datasets/animals_test'))

# Initialize SimplePreprocessor and SimpleDatasetLoader and load data and labels
print('[INFO]: Images loading....')
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=300)
(eval_data, eval_labels) = sdl.load(eval_image_paths, verbose=50)

# Reshape from (3000, 32, 32, 3) to (3000, 32*32*3=3072)
data = data.reshape((data.shape[0], 3072))
eval_data = eval_data.reshape((eval_data.shape[0], 3072))

# Print information about memory consumption
print('[INFO]: Features Matrix: {:.1f}MB'.format(float(data.nbytes / 1024*1000.0)))

# Encode labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)
le_test = LabelEncoder()
eval_labels = le_test.fit_transform(eval_labels)

# Split data into training (75%) and testing (25%) data
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Train and evaluate the k-NN classifier on the raw pixel intensities
print('[INFO]: Classification starting....')

neighbors=[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]
prec=zeros([8,len(neighbors)])
cat_prec=zeros([2,len(neighbors)])
dog_prec=zeros([2,len(neighbors)])
panda_prec=zeros([2,len(neighbors)])
h=0;
for w_ind, weights in enumerate(['uniform', 'distance']):
	for a_ind, algorithm in enumerate(['auto', 'ball_tree', 'kd_tree', 'brute']):
		print('[INFO]: Classification {} / 8'.format(h+1))
		for k_ind, kn in enumerate(neighbors):
			print('[INFO]: Classification for n_neighbors = {}'.format(kn))
			model=KNeighborsClassifier(n_neighbors=kn,	weights=weights,algorithm=algorithm,n_jobs=args['jobs'])
			
			model.fit(train_x, train_y)
			#print(classification_report(test_y, 	model.predict(test_x),target_names=le.classes_))
			rep=classification_report(test_y, model.predict(test_x),
				target_names=le.classes_,output_dict=True)
			prec[h][k_ind-1]=rep['macro avg']['precision']
			if algorithm == 'brute':
				cat_prec[w_ind-1][k_ind-1]=rep['cats']['precision']
				dog_prec[w_ind-1][k_ind-1]=rep['dogs']['precision']
				panda_prec[w_ind-1][k_ind-1]=rep['panda']['precision']
			
		h=h+1
# ****** Figure 1 ******
han=['1','2','3','4','5','6','7','8']
cc=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
c=[cc[0],cc[0],cc[0],cc[0],cc[1],cc[1],cc[1],cc[1]]
marker=[7,4,6,5,7,4,6,5]
label=['uniform:auto', 'uniform:ball_tree', 'uniform:kd_tree', 'uniform:brute','distance:auto', 'distance:ball_tree', 'distance:kd_tree', 'distance:brute']

f1=plt.figure();
for i in range(prec.shape[0]):
	han[i]=plt.scatter(neighbors,prec[i], c=c[i], s=100, marker=marker[i], 	label=label[i])

plt.title('KNN Performance vs k, weight, algorithm');
plt.xlabel('k:number of neighbors');
plt.ylabel('Average Precision');
locs, labels = plt.xticks() 
plt.xticks(neighbors);
plt.legend(handles=han,loc='best')

f1.savefig('figure1.svg')

# ****** Figure 2 ******
han=['1','2','3','4','5','6']
cc=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
c=[cc[0],cc[0],cc[0],cc[1],cc[1],cc[1]];
marker=['o','^','s','o','^','s'];

label=['uniform: Cats', 'uniform: Dogs', 'uniform: Panda','distance: Cats', 'distance: Dogs', 'distance: Panda']

f2=plt.figure();
i=-1
for j in [0,1]:
	i=i+1;
	han[i]=plt.scatter(neighbors,cat_prec[j], c=c[i],s=100,marker=marker[i], label=label[i]);
	
	i=i+1;
	han[i]=plt.scatter(neighbors,dog_prec[j], c=c[i],s=100,marker=marker[i], label=label[i]);

	i=i+1;
	han[i]=plt.scatter(neighbors,panda_prec[j], c=c[i],s=100, marker=marker[i],  label=label[i]);

plt.title('KNN Performance vs k, weight, class');
plt.xlabel('k:number of neighbors');
plt.ylabel('Class Precision');
locs, labels = plt.xticks();
plt.xticks(neighbors);
plt.legend(handles=han,loc='best');
f2.savefig('figure2.svg');

# Evaluate the k-NN classifier on Evaluation Data (eval_data & eval_labels)
print('[INFO]: Evaluation starting....')

neighbors=[2,4,6,8,10,12,14,16,18,20]
prec=zeros([1,len(neighbors)])
cat_prec=zeros([1,len(neighbors)])
dog_prec=zeros([1,len(neighbors)])
panda_prec=zeros([1,len(neighbors)])
h=0;
for w_ind, weights in enumerate(['distance']):
	for a_ind, algorithm in enumerate(['brute']):
		for k_ind, kn in enumerate(neighbors):
			print('[INFO]: Classification for n_neighbors = {}'.format(kn))
			model=KNeighborsClassifier(n_neighbors=kn,	weights=weights,algorithm=algorithm,n_jobs=args['jobs'])
			
			model.fit(train_x, train_y)
			rep=classification_report(eval_labels, model.predict(eval_data),
				target_names=le.classes_,output_dict=True)
			prec[0][k_ind-1]=rep['macro avg']['precision']
			cat_prec[0][k_ind-1]=rep['cats']['precision']
			dog_prec[0][k_ind-1]=rep['dogs']['precision']
			panda_prec[0][k_ind-1]=rep['panda']['precision']

# ****** Figure 3 ******
han=['1','2','3','4']
c=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
marker=['^','s','p','o'];

label=['Cats', 'Dogs', 'Panda','Average']

f3=plt.figure();
i=0
han[i]=plt.scatter(neighbors,cat_prec[0], c=c[i],s=100,marker=marker[i], label=label[i]);

i=i+1
han[i]=plt.scatter(neighbors,dog_prec[0], c=c[i],s=100,marker=marker[i], label=label[i]);

i=i+1
han[i]=plt.scatter(neighbors,panda_prec[0], c=c[i],s=100,marker=marker[i], label=label[i]);

i=i+1
han[i]=plt.scatter(neighbors,prec[0], c=c[i],s=100,marker=marker[i], label=label[i]);

plt.title('KNN Performance vs k & class');
plt.xlabel('k:number of neighbors');
plt.ylabel('Precision');
locs, labels = plt.xticks();
plt.xticks(neighbors);
plt.legend(handles=han,loc='best');
f3.savefig('figure3.svg');
