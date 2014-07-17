import bxPreprocessor as pre
import pickle
import knnMovies as knn

# CHANGE THESE AS NECESSARYs
N = int(input('Enter a value for N : ' ))
seed = int(input('Enter a seed : '))
k = int(input('Enter a value for k (for KNN) : '))
pathToData = "/home/baxte141/BookCrossing/BookCrossingData/AllData/"


print('Reading in data variables...')
bagOfWords = pickle.load(open(pathToData + 'bagOfWords.pkl','rb'), encoding = 'latin1')
booksBestTerms = pickle.load(open(pathToData + 'booksBestTerms.pkl','rb'), encoding = 'latin1')
books = pickle.load(open(pathToData + 'books.pkl','rb'), encoding = 'latin1')
users = pickle.load(open(pathToData + 'users.pkl','rb'), encoding = 'latin1')
ratings = pickle.load(open(pathToData + 'ratings.pkl','rb'), encoding = 'latin1')


temp = pre.splitTestAndTrainGivenN(ratings, N, seed)
trainRatings = temp[0]
validRatings = temp[1]
testRatings = temp[2]
train = pre.convertDataToDataFrame(trainRatings, users, books)
valid = pre.convertDataToDataFrame(validRatings, users, books)
test = pre.convertDataToDataFrame(testRatings, users, books)

authors = train.Author.tolist() + test.Author.tolist() + valid.Author.tolist()
authors = list(set(authors))
publishers = train.Publisher.tolist() + test.Publisher.tolist() + valid.Publisher.tolist()
publishers = list(set(publishers))

print('Building xi_train')
xi_train = pre.build_xi(train, users)
print('Building xj_train')
xj_train = pre.build_xj(train, books)
print('Building wj_train')
wj_train = pre.build_wj(train, books, booksBestTerms, bagOfWords, authors, publishers, dict=True )
print('Building xi_valid')
xi_valid = pre.build_xi(valid, users)
print('Building xj_valid')
xj_valid = pre.build_xj(valid, books)
print('Building wj_valid')
wj_valid = pre.build_wj(valid, books, booksBestTerms, bagOfWords, authors, publishers, dict=True )
print('Building xi_test')
xi_test = pre.build_xi(test, users)
print('Building xj_test')
xj_test = pre.build_xj(test, books)
print('Building wj_test')
wj_test = pre.build_wj(test, books, booksBestTerms, bagOfWords, authors, publishers, dict=True )

mvecs = {}

for isbn in wj_train:
	if isbn not in mvecs.keys():
		mvecs[isbn] = wj_train[isbn]
		
for isbn in wj_test:
	if isbn not in mvecs.keys():
		mvecs[isbn] = wj_test[isbn]
		
for isbn in wj_valid:
	if isbn not in mvecs.keys():
		mvecs[isbn] = wj_valid[isbn]
		
print('Normalizing features')	
numFeatures = len(wj_test[list(wj_test.keys())[0]])
for i in range(0, numFeatures):
	pre.normalizeFeatures(list(mvecs.values()), i, i+1)
	
results = knn.EvaluateKNNRMSE(trainRatings, testRatings, {}, k, mvecs)

print('FINAL RESULTS FOR N = ', N, ', seed = ', seed, ', k = ', k)
results

