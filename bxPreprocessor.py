# TO PREPROCESS DATA:
# ratings = readBookRatings(BX-Book_ratings.csv)
# users = readUserFile(BX-Users.csv)
# books = readBookFile(BX-Books.csv)
#
# removeBooksWithFewRatings(ratings, books, 3)
# bookWords = mapBooksToWordCounts(ratings, books)
# removeInvalidISBNS(ratings, books)
# removeUsersMissingAge(ratings, users)
# removeUsersWithFewRatings(ratings, 6)
# removeLocationsWithFewUsers(ratings, users, 50)
# removeRatingsLowerThan(ratings, 5)
## SYNC UP BOOKWORDS AND USERS
# for uid in bookWords.keys():
#		if uid not in users.keys():
#			del bookWords[uid]
# 
# temp = createBagOfWords(bookWords)
# booksBestTerms = temp[0]
# bagOfWords = temp[1]
# bookVectors = buildBookWordVectors(booksBestTerms, bagOfWords)
# assignUsersAgeRange(users)
# convertFieldToNumeral(users, 0)
# convertFieldToNumeral(books, 1)
# convertFieldToNumeral(books, 3)
# 
# bxdf = convertDataToDataFrame(ratings, users, books)




import htmlStripper;
import urllib.request as urllib
import math
import pickle
import pandas
import numpy
import random

def readBookRatings(filename):
	# Read file, remove column header line.
	f = open(filename, 'r')
	contents = f.readlines()
	f.close()
	del contents[0]

	d = {}
	for line in contents:
		vals = line.replace('\"', '').split(';')
		uid = int(vals[0])
		isbn = vals[1]
		rating = int(vals[2])
		
		if uid not in d.keys():
			d[uid] = {}
		
		d[uid][isbn] = rating
				
	return d
	
def readUserFile(filename):
	f = open(filename, 'r')
	contents = f.readlines()
	f.close()
	del contents[0]

	d= {}
	for line in contents:
		vals = line.replace('\"', '').replace('\n','').split(';')
		# Special cases -- some data points have ';' in the location. . .
		while (len(vals) > 3):
			del(vals[1])
		uid = int(vals[0])
		location = vals[1]
		splitLoc = location.split(', ')
		country = splitLoc[len(splitLoc) - 1]
		age = vals[2]
		if (vals[2] == 'NULL'):
			d[uid] = [ country, -1 ]
		else:
			d[uid] = [ country, int(age) ]
			
	return d

def readBookFile(filename):
	f = open(filename, 'r')
	contents = f.readlines()
	f.close()
	del contents[0]
	
	d = {}
	badCount = 0
	
	for line in contents:
		vals = line.replace('\"', '').replace('&amp;', '').split(';')
		#Find year index.
		isbn = vals[0]
		found = False
		ind = 3
		while (not found) and (ind < len(vals)):
			try:
				year = int(vals[ind])
				found = True
			except ValueError:
				ind = ind + 1
		if (ind >= len(vals)):
			print(isbn)
			continue
				
		isbn = vals[0]
		i = 1
		title = ''
		while (i < ind-1):
			title = title + vals[i]
			i = i + 1
		author = vals[ind -1]
		publisher = vals[ind + 1]
		d[isbn] = [title, author, year, publisher]

	return d
		
def removeInvalidISBNS(ratings, books):
	ct = 0
	toRemove = []
	i = 1
	num = len(ratings.keys())
	for uid in ratings.keys():
		print (i, '/', num, '(so far ', ct, ' will be removed')
		i+=1
		for book in ratings[uid].keys():
			if book not in books.keys():
					toRemove.append(book)
					ct = ct + 1
	i = 1
	num = len(toRemove)
	for isbn in toRemove:
		print (i, '/', num)
		i+=1
		for uid in ratings.keys():
			if isbn in ratings[uid].keys():
				del ratings[uid][isbn]
					
	print(ct, ' invalid ratings removed')

def removeRatingsLowerThan(ratings, minAcceptableRating):
	for uid in ratings.keys():
		toRemove = []
		for isbn in ratings[uid].keys():
			if ratings[uid][isbn] < minAcceptableRating:
				toRemove.append(isbn)
		for isbn in toRemove:
			del ratings[uid][isbn]

def removeUsersWithFewRatings(ratings, users, minAcceptableNumber):
	toRemove = []
	for uid in users.keys():
		if uid not in ratings.keys():
			toRemove.append(uid)
		elif len(ratings[uid].keys()) < minAcceptableNumber:
			toRemove.append(uid)
			
	for uid in toRemove:
		try:
			del ratings[uid]
		except KeyError:
			pass
		del users[uid]
		
def syncResources(bookWords, books):
	toRemove = []
	for uid in bookWords.keys():
		if uid not in users.keys():
			toRemove.append(uid)
			
	for uid in toRemove:
		del bookWords[uid]
		
def removeLocationsWithFewUsers(ratings, userInfo, minAcceptableNumber):
	locationCounts = {}
	usersToCountries = {}
	for uid in userInfo.keys():
		usersToCountries[uid] = userInfo[uid][0]
		
	for country in usersToCountries.values():
		if country in locationCounts.keys():
			locationCounts[country] = locationCounts[country] + 1
		else:
			locationCounts[country] = 1
		
	toRemove = []
	for uid in ratings.keys():
		if (locationCounts[usersToCountries[uid]] < minAcceptableNumber):
			toRemove.append(uid)
			
	for uid in ratings.keys():
		if ((userInfo[uid][0] == ',') or (userInfo[uid][0] == 'n/a')):
			toRemove.append(uid)
			
	for uid in toRemove:
		del ratings[uid]
		del userInfo[uid]
		
def removeUsersMissingAge(ratings, users):
	toRemove = []
	for uid in users.keys():
		if  (users[uid][1] < 0):
			toRemove.append(uid)
	for uid in toRemove:
		if uid in ratings.keys():
			del ratings[uid]
		del users[uid]
	
def removeBooksWithFewRatings(ratings, books, minAcceptableNumber):
	numBookRatings = {}
	for uid in ratings.keys():
		for isbn in ratings[uid].keys():
			if isbn in numBookRatings.keys():
				numBookRatings[isbn] = numBookRatings[isbn] + 1
			else:
				numBookRatings[isbn] = 1
				
	toRemove = []
	for isbn in numBookRatings.keys():
		if (numBookRatings[isbn] < minAcceptableNumber):
			toRemove.append(isbn)
	
	for isbn in books.keys():
		if isbn not in numBookRatings.keys():
			if isbn not in toRemove: #just to be safe
				toRemove.append(isbn)

	# Remove from ratings and books
	numToRemove = len(toRemove)
	i = 1 
	for isbn in toRemove:
		print(i, '/', numToRemove, '  (few ratings)')
		i += 1
		# ratings
		for uid in ratings.keys():
			if isbn in ratings[uid].keys():
				del ratings[uid][isbn]
		# books
		if isbn in books.keys():
			del books[isbn]
			
			
def mapBooksToWordCounts(ratings, books):
	reviewHeader = '<h2>Editorial Reviews</h2>'
	toRemove = []
	d = {}
	i = 1
	num = len(books.keys())
	connErrs = []
	for isbn in books.keys():
		print(i, '/', num)
		i = i + 1
		try:
			url = 'http://www.amazon.com/dp/product-description/' + isbn + '/'
			usocket = urllib.urlopen(url)
			content = usocket.read()
			usocket.close()
			contentStr = content.decode("utf-8", "ignore")
		except urllib.HTTPError:
			print('httperror')
			toRemove.append(isbn)
			continue
		except ConnectionResetError:
			connErrs.append(isbn)
			continue
				
		if reviewHeader not in contentStr:
			toRemove.append(isbn)
			continue
		else:
			parser = htmlStripper.MyHTMLParser()
			parser.feed(contentStr)
			reviews = parser.data.replace('.', ' ').replace('\n',' ').replace('\t',' ').replace(',',' ').replace('\"',' ').replace('\'',' ').replace('-',' ').replace('(',' ').replace(')',' ') \
				.replace('	','').replace('\t', ' ').replace('*',' ').replace('\\',' ').replace('?',' ').replace('!',' ').replace(';', ' ').replace(':', ' ').lower()
			words = reviews.split(' ')
			words = [value for value in words if value != '']
			d[isbn] = {}
			for word in words:
				if word in d[isbn].keys():
					d[isbn][word] = d[isbn][word] + 1
				else:
					d[isbn][word] = 1
	
	for isbn in toRemove:
		if isbn in books.keys():
			del books[isbn]
		for uid in ratings.keys():
			if isbn in ratings[uid].keys():
				del ratings[uid][isbn]
	
	return d
			
def removeBooksWithoutReviews(ratings, books):
	reviewHeader = '<h2>Editorial Reviews</h2>'
	toRemove = []
	d = {}
	i = 0
	num = 0
	for isbn in books.keys():
		num += 1
		url = 'http://www.amazon.com/dp/product-description/' + isbn + '/'
		usocket = urllib.urlopen(url)
		content = usocket.read()
		usocket.close()
		contentStr = content.decode("utf-8", "ignore")
		if reviewHeader not in contentStr:
			toRemove.append(isbn)
			i+=1
		print(i,'/',num,' seen so far will be removed.')
			
	for isbn in toRemove:
		if isbn in books.keys():
			del books[isbn]
		for uid in ratings.keys():
			if isbn in ratings[uid].keys():
				del ratings[uid][isbn]


def calcTF(bookWordMap, term, isbn):
	try:
		termAppearances = bookWordMap[isbn][term]
	except KeyError:
		return -1
	totalTerms = 0
	for v in bookWordMap[isbn].values():
		totalTerms = totalTerms + v
	return termAppearances/totalTerms
	
def calcIDF(bookWordMap, term):
	numDocs = len(bookWordMap.keys())
	numDocsWithTerm = 0
	for bookID in bookWordMap.keys():
		if term in bookWordMap[bookID].keys():
			numDocsWithTerm = numDocsWithTerm + 1
			
	return math.log( numDocs / numDocsWithTerm, math.e )
	
def calcTFIDF(bookWordMap, term, isbn):
	return calcTF(bookWordMap, term, isbn)*calcIDF(bookWordMap, term)
	
def calcTFIDFWithIDFDict(bookWordMap, term, isbn, idfs):
	return calcTF(bookWordMap, term, isbn)*idfs[term]

def getTopWords(bookWordMap, isbn):
	if isbn not in bookWordMap.keys():
		return []
	numTerms = min( 70, len(bookWordMap[isbn].keys()))
	topTerms = []
	
	tfidfs = {}
	for w in bookWordMap[isbn].keys():
		score = calcTFIDF(bookWordMap, w, isbn)
		if score in tfidfs.keys():
			tfidfs[score].append(w)
		else:
			tfidfs[score] = [ w ]
	
	scores = sorted(list(tfidfs.keys()), reverse=True)
	for s in scores:
		words = tfidfs[s]
		for w in words:
			topTerms.append(w)
			
		if (len(topTerms) > numTerms):
			topTerms[numTerms:len(topTerms)] = []
		if (len(topTerms) == numTerms):
			break
			
	return topTerms

def getTopWordsWithIDFDict(bookWordMap, isbn, idfs):
	if isbn not in bookWordMap.keys():
		return []
	numTerms = min( 70, len(bookWordMap[isbn].keys()))
	topTerms = []
	
	tfidfs = {}
	for w in bookWordMap[isbn].keys():
		score = calcTFIDFWithIDFDict(bookWordMap, w, isbn, idfs)
		if score in tfidfs.keys():
			tfidfs[score].append(w)
		else:
			tfidfs[score] = [ w ]
	
	scores = sorted(list(tfidfs.keys()), reverse=True)
	for s in scores:
		words = tfidfs[s]
		for w in words:
			topTerms.append(w)
			
		if (len(topTerms) > numTerms):
			topTerms[numTerms:len(topTerms)] = []
		if (len(topTerms) == numTerms):
			break
			
	return topTerms
	
	
def createBagOfWords(bookWords):
	num = len(bookWords)
	i = 1
	idfs = {}
	for isbn in bookWords.keys():
			print(i,'/',num)
			i += 1
			for word in bookWords[isbn].keys():
				if word not in idfs.keys():
					idfs[word] = calcIDF(bookWords, word)
					
	
	booksBestTerms = {}
	i = 1
	print('Finding all books best terms')
	for isbn in bookWords.keys():
		print(i,'/',num)
		i += 1
		booksBestTerms[isbn] = getTopWordsWithIDFDict(bookWords, isbn, idfs)
		
	print('Creating bag of words')
	bagOfWords = []
	for isbn in booksBestTerms.keys():
		print(i,'/',num)
		i += 1
		bagOfWords = bagOfWords + booksBestTerms[isbn]
		bagOfWords = list(set(bagOfWords)) #Removes duplicates
		
	return [booksBestTerms, bagOfWords]
		

def defineAgeRanges(users):
	numUsers = len(users.keys())
	tenthUsers = math.floor(numUsers/10)
	age = 1
	ageCutoffs = []
	chunk = 1
	usersSeen = 0
	while (chunk < 10):
		for uid in users.keys():
			if (users[uid][1] == age):
				usersSeen += 1
			if (usersSeen >= chunk*tenthUsers):
				ageCutoffs.append(age)
				chunk += 1
		age += 1
	return ageCutoffs
	
def assignUsersAgeRange(users):
	ageRanges = defineAgeRanges(users)
	newUsersAges = {}
	for uid in users.keys():
		if users[uid][1] > ageRanges[8]:
			newUsersAges[uid] = 10
	for i in reversed(range(0,9)):
		for uid in users.keys():
			if users[uid][1] <= ageRanges[i]:
				newUsersAges[uid] = i + 1
				
	for uid in users.keys():
		users[uid][1] = newUsersAges[uid]
		
	for uid in users.keys():
		users[uid][1] = users[uid][1] - 5.5
		
def convertFieldToNumeral(d, fieldInd):
	vals = []
	for k in d.keys():
		vals.append(d[k][fieldInd])
	vals = list(set(vals)) # removes duplicates
	for k in d.keys():
		d[k][fieldInd] = vals.index(d[k][fieldInd])

def convertDataToDataFrame(ratings, users, books):
	dfratings = []
	dfuids = []
	dfages = []
	dfcountries = []
	dfisbns = []
	dftitles = []
	dfauthors = []
	dfpublishers = []
	dfyears = []
	for uid in ratings.keys():
		for isbn in ratings[uid].keys():
			dfratings.append(ratings[uid][isbn])
			dfuids.append(uid)
			dfages.append(users[uid][1])
			dfcountries.append(users[uid][0])
			dfisbns.append(isbn)
			dftitles.append(books[isbn][0])
			dfauthors.append(books[isbn][1])
			dfpublishers.append(books[isbn][3])
			dfyears.append(books[isbn][2])
			
	d = {'Rating' : pandas.Series(dfratings), 'UserID' : pandas.Series(dfuids), 'Age' : pandas.Series(dfages), 'Country' : pandas.Series(dfcountries), \
	'ISBN' : pandas.Series(dfisbns), 'Title':pandas.Series(dftitles), 'Author' : pandas.Series(dfauthors), 'Publisher' : pandas.Series(dfpublishers), 'Year' : pandas.Series(dfyears) }
	
	df = pandas.DataFrame(d)
	return df
			
def buildBookWordVectors(booksBestTerms, bagOfWords):
	booksVectors = {}
	numBooks = len(booksBestTerms.keys())
	vecLength = len(bagOfWords)
	ct = 1
	print('Calculating initial vectors')
	for isbn in booksBestTerms.keys():
		#print(ct, '/', numBooks)
		ct += 1
		booksVectors[isbn] = [0] * vecLength
		for w in booksBestTerms[isbn]:
			booksVectors[isbn][bagOfWords.index(w)] = 1
			
	#normalize
	
	for i in range(0, vecLength):
		print (i+1,'/',vecLength)
		total = 0
		for isbn in booksVectors.keys():
			total = total + booksVectors[isbn][i]
		avg = total/numBooks
		if (avg == 0):
			booksVectors[isbn][i] = 0
			continue

		dev = 0
		for isbn in booksVectors.keys():
			dev = dev + (booksVectors[isbn][i] - avg)**2
		dev = dev/numBooks
		dev = math.sqrt(dev)
		
		for isbn in booksVectors.keys():
			booksVectors[isbn][i] = (booksVectors[isbn][i] - avg)/dev
		
	return booksVectors

def calcBookVecsInRange(lowj, highj, booksBestTerms, bagOfWords):
	bbtKeys = list(booksBestTerms.keys())
	for j in range(lowj,highj):
		print ('On chunk', j, '/', highj)
		lower = j*1000
		upper = j*1000 + 1000
		keys = []
		for i in range(lower, upper):
				keys.append(bbtKeys[i])
		bbt = {}
		for k in keys:
				bbt[k] = booksBestTerms[k]
		filename = 'C:/Users/Daniel/My Documents/Daniel\'s Homework/Research/'\
				+ 'BookCrossing/Order2/bvecs/' + str(lower) + '-' \
				+ str(upper) + '.pkl'
		f = open(filename, 'wb')
		bvecs = buildBookWordVectors(bbt, bagOfWords)
		pickle.dump(bvecs, f)
		f.close()
		del bvecs
		bvecs = {}
		
def buildSingleBookWordVector(booksBestTerms, books,  authors, publishers, bagOfWords, isbn):
	vecLength = len(bagOfWords) + len(authors) + len(publishers)
	bookVector = [0] * vecLength
	for w in booksBestTerms[isbn]:
		bookVector[bagOfWords.index(w)] = 1
	authorInd = authors.index(books[isbn][2])
	publisherInd = publishers.index(books[isbn][3])
	bookVector[len(bagOfWords)  + authorInd] = 1
	bookVector[len(bagOfWords) + len(authors) + publisherInd] = 1
	return bookVector

def normalizeFeaturesDict(data, minFeature = 0, maxFeature = -1):
	numFeatures = len(data[list(data.keys())[0]])
	numPoints = len(data)
	if maxFeature < 0:
		maxFeature = numFeatures
	featureTotals = []
	for i in range(0, numFeatures):
		featureTotals.append(0)

	print('Summing Feature Vals . . .')
	for i in data.keys():
		for j in range(minFeature, maxFeature):
			featureTotals[j] += data[i][j]
			
			
	print('Calculating Feature Means . . .')
	featureMeans = [];
	for i in range(0, numFeatures):
		featureMeans.append(featureTotals[i] / numPoints);
	
	print('Calculating Feature Standard Deviations . . .')	
	featureDevs = []
	for j in range(0, numFeatures):
		featureDevs.append(0);
	
	for i in data.keys():
		for j in range(minFeature, maxFeature):
			featureDevs[j] += ( data[i][j] - featureMeans[j] )**2
			
	for j in range(minFeature, maxFeature):
		featureDevs[j] = math.sqrt( featureDevs[j] / numPoints )
			
	print('Normalizing Data . . .')
	newData = {}
	for i in data.keys():
		newData[i] = []
		for j in range(0, numFeatures):
			newData[i].append(data[i][j])

		
		
	for i in data.keys():
		for j in range(minFeature, maxFeature):
			newData[i][j] = (newData[i][j] - featureMeans[j]) / featureDevs[j]

	return newData
	
	
def splitTestAndTrainGivenN(ratings, N, seed):
	#First, make copy of  ratings which includes only users with enough ratings for test and validation sets
	validUsers = []
	for uid in ratings.keys():
		# 2*N for train and validation, at least 10 for test.
		if len(ratings[uid].keys()) < (N+20):
			continue
		else:
			validUsers.append(uid)
		
	# Seed random number generator
	random.seed(seed)
		
	train = {}
	validation = {}
	test = {}
	#For each user
	num = len(validUsers)
	for uid in validUsers:
		keys = list(ratings[uid].keys())
		#Make train data
		train[uid] = {}
		while len(train[uid].keys()) < N:
			ind = random.randint(0, len(keys) - 1)
			key = keys[ind]
			train[uid][key] = ratings[uid][key]
			keys.remove(key)
			
		#Make validation data
		validation[uid] = {}
		while len(validation[uid].keys()) < 10:
			ind = random.randint(0,len(keys) - 1)
			key = keys[ind]
			validation[uid][key] = ratings[uid][key]
			keys.remove(key)
			
		#Make test data (all remaining keys)
		test[uid] = {}
		for key in keys:
			test[uid][key] = ratings[uid][key]
			
	result = [train,  validation, test]
	return result
	
		

def splitTestAndTrainFixedRate(ratings):
	train = {}
	test = {}
	for uid in ratings.keys():
		train[uid] = {}
		test[uid] = {}
		i = 0
		for b in ratings[uid].keys():
			if (i%2) :
				train[uid][b] = ratings[uid][b]
			else:
				test[uid][b] = ratings[uid][b]
			i = i + 1
				
	result = [train, test]
	return result
	
def build_xi(data, users):
	userIDs = data.UserID.unique().tolist()
	xi = []
	for uid in userIDs:
		xi.append(users[uid])
	
	return numpy.asarray(xi)
	
def build_xj(data, books):
	bookIDs = data.ISBN.unique().tolist()
	xj = []
	for b in bookIDs:
		xj.append(books[b][1:4])
		
	return numpy.asarray(xj)

def build_wj(data, books, booksBestTerms, bagOfWords):
	bookIDs = data.ISBN.unique().tolist()
	wj = []
	authors = data.Author.unique().tolist()
	publishers = data.Publisher.unique().tolist()
	#i = 1
	#num = len(bookIDs)
	for isbn in bookIDs:
		#print(i,'/',num)
		#i += 1
		wj.append(buildSingleBookWordVector(booksBestTerms, books,  authors, publishers, bagOfWords, isbn))
	return numpy.asarray(wj)
