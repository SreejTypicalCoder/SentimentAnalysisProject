from simpleDemo import NBClassifier,MaxEnt
from simpleDemo import processTweet, extract_features,getFeatureVector, getStopWordList
stopWords = getStopWordList('data/feature_list/stopwords.txt')
# Test the classifier
testTweet = 'Congrats'
processedTestTweet = processTweet(testTweet)
#sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
sentiment = MaxEnt.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
print ("testTweet = %s \nsentiment = %s\n" % (testTweet , sentiment))