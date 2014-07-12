import bxPreprocessor as pre
import pickle

# CHANGE THESE AS NECESSARYs
N = 10
seed = 1
pathToData = "C:/Users/Daniel/Dropbox/Yahoo-project-preprocessing_scripts/BookCrossing/AllData/"


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
print('Building xi_train')
xi_train = pre.build_xi(train, users)
print('Building xj_train')
xj_train = pre.build_xj(train, books)
print('Building wj_train')
wj_train = pre.build_wj(train, booksBestTerms, bagOfWords)
print('Building xi_valid')
xi_valid = pre.build_xi(valid, users)
print('Building xj_valid')
xj_valid = pre.build_xj(valid, books)
print('Building wj_valid')
wj_valids = pre.build_wj(valid, booksBestTerms, bagOfWords)
print('Building xi_test')
xi_test = pre.build_xi(test, users)
print('Building xj_test')
xj_test = pre.build_xj(test, books)
print('Building wj_test')
wj_test = pre.build_wj(test, booksBestTerms, bagOfWords)

print('Normalizing wj vectors')
pre.normalizeFeatures(wj_train)
pre.normalizeFeatures(wj_test)
pre.normalizeFeatures(wj_valid)

