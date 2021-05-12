import sys
import getopt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import random




		


def banner():
	print ('\n\n*******************************************************************************************')
	print ('*            															  ')
	print ('*   Web Application Firewall using Machine Learning and Features Engineering    ')
	print ('*                                             							  ')
	print ('*   Aref Shaheed, Mhd Bassam Kurdy              							  ')
	print ('*                                             							  ')
	print ('*   Syrian Virtual University - Master of Web Technologies - WPR - S17	  ')
	print ('*            															  ')
	print ('*******************************************************************************************\n')


def usage():
	print ('\n\n*******************************************************************************************')
	print ('*            															  ')
	print ("*   Usage:")
	print ("*   \t-d: dataset   (1 for HTTPParams, 2 for CSIC 3 for both)")
	print ("*   \t-a: algorithm (1 for Naive Bayes, 2 for Logistic Regression, 3 for Decision Tree)")
	print ('*            															  ')
	print ("*   example: waf_training.py -d 1 -a 2")
	print ('*            															  ')
	print ('*            															  ')
	print ('*******************************************************************************************\n')	


def shuffls_dataset(option):
	option = int(option)
	if(option == 1):
		fid = open("HTTPParams.csv", "r")
		li = fid.readlines()
		header, rest=li[0], li[1:]
		fid.close()
		random.shuffle(rest)
		fid = open("HTTPParams_shuffled.csv", "w")
		new_lines = [header]+rest
		fid.writelines(new_lines)
		fid.close()
	elif(option == 2):
		fid = open("CSIC.csv", "r")
		li = fid.readlines()
		header, rest =li[0], li[1:]
		fid.close()
		random.shuffle(rest)
		fid = open("CSIC_shuffled.csv", "w")
		new_lines = [header]+rest
		fid.writelines(new_lines)
		fid.close()	
	elif(option == 3):
		fid = open("CSIC_HTTPParams.csv", "r")
		li = fid.readlines()
		header, rest =li[0], li[1:]
		fid.close()
		random.shuffle(rest)
		fid = open("CSIC_HTTPParams_shuffled.csv", "w")
		new_lines = [header]+rest
		fid.writelines(new_lines)
		fid.close()
	else:
		return False
	
	return True

def choose_algorithm(option):
	global algorithm
	option = int(option)
	if(option == 1):
		print('[+] \t Algorithm : Naive Bayes \n')
		from sklearn.naive_bayes import GaussianNB
		algorithm = GaussianNB()
	elif(option == 2):
		print('[+] \t Algorithm : Logistic Regression \n')
		from sklearn.linear_model import LogisticRegression
		algorithm = LogisticRegression()
	elif(option == 3):
		print('[+] \t Algorithm : Decision Tree  \n')
		from sklearn.tree import DecisionTreeClassifier
		algorithm = DecisionTreeClassifier()
	else:
		return False
		
	return True



def choose_dataset(option):
	global dataset
	option = int(option)
	if(option == 1):
		print('[+] \t Dataset : HTTPParams 2015\n')
		shuffls_dataset(option)
		dataset_file = 'HTTPParams_shuffled.csv'
		col_names = ['payloads', 'payload_len','alpha','non_alpha','attack_feature','label']
	elif(option == 2):
		print('[+] \t Dataset : CSIC 2010\n')
		shuffls_dataset(option)
		dataset_file = 'CSIC_shuffled.csv'
		col_names = ['method', 'url', 'payloads', 'payload_len','alpha','non_alpha','attack_feature','label']
	elif(option == 3):
		print('[+] \t Dataset : CSIC 2010 & HTTPParams 2015\n')
		shuffls_dataset(option)
		dataset_file = 'CSIC_HTTPParams.csv'
		col_names = ['payload_len','alpha','non_alpha','attack_feature','label']
	else:
		return False
		
	dataset = pd.read_csv(dataset_file, header=None, names=col_names, skiprows = 1)
	# print(dataset)
	# dataset = shuffle(dataset)
	# print(dataset)

	return True



def train(ds, al):
	check_if_ds_selected = choose_dataset(ds)
	check_if_al_selected = choose_algorithm(al)

	if(not check_if_ds_selected or not check_if_al_selected):
		print('check your input by reading usage below !!')
		usage()
		exit()

	global algorithm
	global dataset
	feature_cols = ['payload_len','alpha','non_alpha','attack_feature']

	X = dataset[feature_cols]
	y = dataset.label

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
	algorithm.fit(X_train, y_train)
	y_pred_class = algorithm.predict(X_test)
	algorithm.fit(X_train, y_train)



	print('[+] \t Classification accuracy : ', "{:.2f}".format(metrics.accuracy_score(y_test, y_pred_class) * 100), '%\n')


	print('[+] \t Percentage of Anomaly requests in test set : ', "{:.2f}".format(y_test.mean()*100), '%\n' )
	print('[+] \t Percentage of Normal requests in test set : ', "{:.2f}".format((1 - y_test.mean()) * 100), '%\n' )






	confusion = metrics.confusion_matrix(y_test, y_pred_class)
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]

	print('[+] \t TP : ', TP , 'from ', (TP+TN+FP+FN) )
	print('    \t TN : ', TN , 'from ', (TP+TN+FP+FN) )
	print('    \t FP : ', FP , 'from ', (TP+TN+FP+FN) )
	print('    \t FN : ', FN , 'from ', (TP+TN+FP+FN) )


	print('\n[+] \t Metrics : ')
	print('\t[-]  Accuracy Score (train_test_split): ', "{:.2f}".format(metrics.accuracy_score(y_test, y_pred_class) * 100), '%')
	print('\t[-]  Accuracy Score (k-fold): ', "{:.2f}".format(cross_val_score(algorithm, X, y, cv=100, scoring='accuracy').mean() * 100), '%')
	
	print('\t[-]  Classification Error : ', "{:.2f}".format((1- metrics.accuracy_score(y_test, y_pred_class)) * 100), '%')
	print('\t[-]  Sensitivity : ', "{:.2f}".format(metrics.recall_score(y_test, y_pred_class) * 100), '%')
	specificity = TN / (TN + FP)
	print('\t[-]  Specificity : ', "{:.2f}".format(specificity * 100), '%')
	false_positive_rate = FP / float(TN + FP)
	print('\t[-]  False Positive Rate : ', "{:.2f}".format(false_positive_rate * 100), '%')
	precision = TP / float(TP + FP)
	print('\t[-]  Precision : ', "{:.2f}".format(precision * 100), '%')


def start(argv):
	banner()
	if len(sys.argv) < 5:
		usage()
		sys.exit()
	try:
		opts, args = getopt.getopt(argv, "a:d:")
	except getopt.GetoptError:
		print ("Error in arguments")
		usage()
		sys.exit()

	for opt, arg in opts:
		if opt == '-a':
			algorithm = arg
		elif opt == '-d':
			dataset = arg

	try:
		train(dataset, algorithm)
	except Exception as e:
		print('Exception !!')
		print(e)


if __name__ == "__main__":
	try:
		start(sys.argv[1:])
	except KeyboardInterrupt:
		print ("KeyboardInterrupt Exception !! Bye :)")



