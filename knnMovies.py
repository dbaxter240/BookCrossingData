from heapq import nlargest
from numpy import argmax
from numpy import sqrt
import numpy as np
from math import floor, fabs, log
import collections

def EvaluateKNNPrecAtK(UtoM_train, UtoM_test, simMat, k, precAt, minTrainMovs = 1, minTestMovs = 1):
	total = 0
	numUsersEvaluated = 0;
	numUsers = len(UtoM_test.keys());
	for uID in UtoM_test.keys():
		#print('Evaluating user',(numUsersEvaluated+1),'/',numUsers, '    uid : ', uID);
		if (uID not in UtoM_train.keys()):
			continue;
		if (len(UtoM_train[uID].keys()) < minTrainMovs):
			continue;
		if (len(UtoM_test[uID].keys()) < minTestMovs):
			continue;
		
		#else
		numUsersEvaluated += 1;
		movsToRank = list(UtoM_test[uID].keys());	
		actRanks = UtoM_test[uID];
		
		predRanks = classifyMovies(movsToRank, UtoM_train[uID], simMat, k);
		prec = CalcPrecisionAtK(predRanks, actRanks, precAt);
		total = total + prec;
		

	precAtK = total / numUsersEvaluated
	return precAtK;
	
	
def EvaluateKNNNDCG(UtoM_train, UtoM_test, simMat, k, ndcg_at, minTrainMovs = 1, minTestMovs = 1):
	total = 0
	numUsersEvaluated = 0;
	numUsers = len(UtoM_test.keys());
	for uID in UtoM_test.keys():
		#print('Evaluating user',(numUsersEvaluated+1),'/',numUsers, '    uid : ', uID);
		if (uID not in UtoM_train.keys()):
			continue;
		if (len(UtoM_train[uID].keys()) < minTrainMovs):
			continue;
		if (len(UtoM_test[uID].keys()) < minTestMovs):
			continue;
		
		#else
		numUsersEvaluated += 1;
		movsToRank = list(UtoM_test[uID].keys());	
		actRanks = UtoM_test[uID];
		predRanks = classifyMovies(movsToRank, UtoM_train[uID], simMat, k);
		
		# get movie presentation order for actual and predicted rankings.
		actMovies = list(reversed(sorted(actRanks, key=actRanks.get)))
		predMovies = list(reversed(sorted(predRanks, key=predRanks.get)))
		predMoviesActRanks = []
		for m in predMovies:
			predMoviesActRanks.append(actRanks[m])
		
		ndcg = CalcNDCG(predMoviesActRanks, ndcg_at)
		total = total + ndcg;
		

	avgNDCG = total / numUsersEvaluated
	return avgNDCG;
		
		
# Optional arg minTrainMovs 
def EvaluateKNNRMSE(UtoM_train, UtoM_test, simMat, k, mVecs, minTrainMovs = 1):
	""" Classifies movies for every user in test data using classifyMovies function
	and computes RMSE of classifications on each user. Returns a list of two elements --
	the average RMSE value across all users and the dictionary of userIDs to each
	users individually calculated RMSE."""
	RMSEs = {};
	MAEs = {};
	numUsers = len(UtoM_test.keys());
	i = 1;
	totalRMSE = 0;
	totalMovies = 0;
	for uID in UtoM_test.keys():
		print('Evaluating user',i,'/',numUsers, '    uid : ', uID);
		i += 1;
		if (uID not in UtoM_train.keys()):
			#no data on user, on to next user
			continue;
		if (len(UtoM_train[uID].keys()) < minTrainMovs):
			continue;
			
		#else
		movsToRank = list(UtoM_test[uID].keys());

		predRanks = classifyMovies(movsToRank, UtoM_train[uID], simMat, k, mVecs);
		actRanks = UtoM_test[uID];
		userRMSE = CalcRMSE(predRanks, UtoM_test[uID]);
		totalRMSE += (userRMSE**2)*len(UtoM_test[uID]);
		totalMovies += len(UtoM_test[uID]);
		RMSEs[uID] = userRMSE;
		userMAE = CalcMAE(predRanks, actRanks);
		MAEs[uID] = userMAE;
		
		
	avgRMSE = 0;
	for uID in RMSEs.keys():
		avgRMSE += RMSEs[uID];
	avgRMSE = avgRMSE / len(RMSEs.keys());
	
	avgMAE = 0;
	for uID in MAEs.keys():
		avgMAE += MAEs[uID];
	avgMAE = avgMAE / len(MAEs.keys());
	
	usersEvaluated = len(RMSEs.keys());
	totalRMSE = sqrt(totalRMSE / totalMovies)
	return [avgRMSE, totalRMSE, avgMAE, usersEvaluated];




def classifyMovies(movsToRank, movsRanked, simMat, k, mVecs=[]):
	""" Uses KNN alg. to classify the provided set of movies based
	on the movies the user has already ranked. """
	rankings = {};
	numToRank = len(movsToRank);
	i = 1;
	for mToRank in movsToRank:
		#print('    ranking movie',i,'/',numToRank);
		i += 1;
		sims = {};
		for m in movsRanked.keys():
			# Calculate similarity of each movie ranked to the one
			# we are trying to rank.

			#similarity = GetValFromSimMat(simMat, m, mToRank);
			similarity = CalculateSimilarity(mVecs[m], mVecs[mToRank]);
			if similarity in sims.keys():
				sims[similarity].append(m);
			else:
				sims[similarity] = [m];
			
			#mostSimilar list of most similar movie IDs.
			mostSimilar = GetMostSimilarMovies(sims, k);
			
			# set classification to the weighted sum of the k nearest.
			classification = 0
			for mov in mostSimilar:
				r = movsRanked[mov]
				classification += r
				
			classification = classification / len(mostSimilar)
			# Count the number of times each ranking shows up in k most similar
			# movies, and classify the movie accordingly.
			#common_classes = [0, 0, 0, 0, 0];
			#for mov in mostSimilar:
			#	r = movsRanked[mov];
			#	common_classes[r-1] += 1;

			#classification = argmax(common_classes)+1;
			
			# End for over movsRanked
			
		rankings[mToRank] = classification;
		
		# End for over movsToRank.
			
	return rankings;
			
		
def CalcRMSE(predRanks, actRanks):
	
	diff_sum = 0;
	for mID in predRanks.keys():
		square_diff = (predRanks[mID] - actRanks[mID])**2;
		diff_sum += square_diff;
	if (len(predRanks.keys()) == 0):
		diff_sum = 0;
	else:
		diff_sum = diff_sum / len(predRanks.keys());
	diff_sum = sqrt(diff_sum);
	
	return diff_sum;
	

def CalcPrecisionAtK(predRanks, actRanks, k):
	retreived = GetTopKKeys(predRanks, k);
	k = min(k, len(actRanks))
	numSeen = 0;
	numGood = 0;
	# For rank in 5,4,3,2,1...
	for rank in reversed(range(1,6)):
		# Get all movies that were labelled with rank.
		movs = GetMovsOfRank(actRanks, rank);
		
		#Find how many movies that were given the current rank are
		# in the predicted movies.
		intersect = list(set(retreived) & set(movs));
		
		# Add to the number of good values, but no higher than the
		# value of k, and no more than the number of movies missed of
		# higher rank.
		numGood += min(k - numSeen, k - numGood, len(intersect));
		numSeen += len(movs);
		if (numSeen >= k):
			break;
	precAtK = numGood/ k
	#print(precAtK);
	return precAtK;
	
	
def CalcNDCG(ranks, pos):
	dcg = CalcDCG(ranks, pos)
	ideal = sorted(ranks, reverse = True)
	idcg = CalcDCG(ideal, pos)
	return dcg/idcg;
	
def CalcDCG(ranks, pos):
	upper = min(pos, len(ranks))
	score = ranks[0]
	for i in range(1, upper):
		score = score + ranks[i] / log( i+2, 2)
		
	return score
	

#For testing purposes.
def CalcMAE(predRanks, actRanks):
	diff_sum = 0;
	for mID in predRanks.keys():
		abs_diff = fabs(predRanks[mID] - actRanks[mID]);
		diff_sum += abs_diff;
		
	diff_sum = diff_sum / len(predRanks.keys());
	diff_sum = sqrt(diff_sum);
	
	return diff_sum;
	
	
def CalculateSimilarity(v1, v2):
	similarity = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2));
	return similarity;
	
	
#UTILITY FUNCTIONS
def MapUsersToMovies(data, genreInfo):
	""" Given train or test panda data in the format of the Movie Lens
	data, return a dictionary mapping all userIDs to a list of movies
	they watched and the rankings for those movies (a list of two 
	element lists."""
	dictionary = {};
	usersSeen = [];
	num_reviews = len(data);
	for i in range(0, num_reviews-1):
		uID = data.iloc[i, 3];
		mID = data.iloc[i, 1];
		rating = data.iloc[i,4];
		if uID not in usersSeen:
			usersSeen.append(uID);
			dictionary[uID] = {};
		if mID not in dictionary[uID].keys():
			dictionary[uID][mID] = rating;
		
	return dictionary;


def GetValFromSimMat(simMatrix, mID1, mID2):
	""" Retrieves similarity of two movies from similarity matrix
	calculated in CalcSimilarityMatrix function. This approach is used
	to prevent having to recalculate similarities of high dimensional
	vectors several times for the same movies."""
	d = simMatrix[mID1];
	if mID2 in d.keys():
		return d[mID2];
	else:
		d = simMatrix[mID2];
		return d[mID1];
		
		
def CalcSimilarityMatrix(mVecs):
	""" Provided a mapping from movie IDs to movie feature vectors,
	this function calculates a dictionary of similiarties between all
	possible pairs of movies (using cosine similarity.) To retrieve
	similarities from the calculated similarity matrix, use the
	GetValFromSimMat function."""
	
	mat = {};
	toCompare = list(mVecs.keys());
	numMovs = len(mVecs);
	i=0;
	for m1 in mVecs.keys():
		print('Similarity Matrix row', i, '/',numMovs);
		i = i+1;
		sims = {};
		for m2 in toCompare:
			sims[m2] = CalculateSimilarity(mVecs[m1], mVecs[m2]);
			
		mat[m1] = sims;
		toCompare.remove(m1);
		
	return mat;
	
	
def GetMostSimilarMovies(similarities, k):
	"""Gets the k best movies from a dictionary mapping similarity values
	to movies. This function is meant to operate on a dictionary as generated
	in the 'classifyMovies' function."""
	similarityVals = sorted(list(similarities.keys()), reverse=True);
	mostSimilar = [];
	for simVal in similarityVals:
		movies = similarities[simVal];
		for m in movies:
			mostSimilar.append(m);
		# If we added a few too many movies, trim the last few 
		# we added off so our list is of length k.
		if (len(mostSimilar) > k):
			mostSimilar[k:len(mostSimilar)] = [];
		if (len(mostSimilar) == k):
			break;
		
	return mostSimilar;
	
	
def ReduceDataToNUsers(UtoM_train, UtoM_test, n):
	reduced_train = {};
	reduced_test  = {};
	count = 0;
	for uID in UtoM_train.keys():
		if uID in UtoM_test.keys():
			reduced_train[uID] = UtoM_train[uID];
			reduced_test[uID]  = UtoM_test[uID];
			count = count + 1;
		if (count == n):
			break;
	return [reduced_train, reduced_test];

def listDiff(a, b):
    b = set(b)
    return [aa for aa in a if aa not in b]


### Cross Validation Functions ###

def SplitDataIntoNFolds(UtoM, numFolds):
	folds = [];
	for i in range(0, numFolds):
		folds.append({});
	for uID in UtoM.keys():
		#Split this user's movies into folds
		movies = UtoM[uID];
		partitioned = [];
		for i in range(0, numFolds):
			partitioned.append({});
		ind = 0;
		for m in UtoM[uID].keys():
			partitioned[ind][m] = UtoM[uID][m];
			ind = (ind + 1)%numFolds;
		
		for i in range(0, numFolds):
			folds[i][uID] = partitioned[i]; 
			
	return folds;
		
		
def GetAverageError(folds, k, simMat):
	numFolds = len(folds);
	errors = [];
	for i in range(0, numFolds):
		trainData = folds[i];
		testData = {};
		for uID in trainData.keys():
			testData[uID] = {};
		for j in listDiff(range(0,numFolds), [i]):
			fold = folds[j];
			for uID in fold.keys():
				for m in fold[uID].keys():
					testData[uID][m] = fold[uID][m];
		e = EvaluateKNN(trainData, testData, simMat, k);
		errors.append(e[0]);
		
	avgErr = 0;
	for err in errors:
		avgErr += err;
	avgErr = avgErr / len(errors);
	
	return [avgErr, errors]		
		
				
def EvaluateKs(folds, ks, simMat):
	kErrs = {};
	for k in ks:
		l = GetAverageError(folds, k, simMat);
		err = l[0];
		kErrs[k] = err;
		
	return kErrs;
	
def GetTopKKeys(d, k):
	d = collections.Counter(d);
	if (k >= len(d.keys())):
		return list(d.keys());
		
	#else
	top = [];
	for k, v in d.most_common(k):
		top.append(k);
	return top;
	
def GetMovsOfRank(movs, rank):
	l = [];
	for m in movs.keys():
		if (movs[m] == rank):
			l.append(m);
			
	return l;
	
def MapMovieVectors(train, test, train_vectors, test_vectors, movieGenres):
	d = {};
	movieIDs = train.movieID.unique().tolist();
	j = 0;
	for id0 in movieIDs:
		genreInfo = movieGenres.iloc[list(movieGenres.movieID).index(id0)]
		mGenre = []
		for j in range(3, 28):
			mGenre.append(genreInfo[j])
		d[id0] = mGenre + []#list(train_vectors[j]);
		j = j+1;
		
	movieIDs = test.movieID.unique().tolist();
	j = 0;
	for id0 in movieIDs:
		genreInfo = movieGenres.iloc[list(movieGenres.movieID).index(id0)]
		mGenre = []
		for j in range(3, 28):
			mGenre.append(genreInfo[j])
		if id0 not in d.keys():
			d[id0] = mGenre + []#+ list(test_vectors[j]);
		j = j+1;
	
	return d;
