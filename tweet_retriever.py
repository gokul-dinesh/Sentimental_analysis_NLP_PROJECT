import tweepy
import pandas as pd
import csv
import re
import string
import preprocessor as p

consumer_key = 'ae99XbZgQbS7HiuFOXlj5vIN6'
consumer_secret = 'd3EjfvrDWUUF1uHfAE0BKhqds24yajh0Is9PM1ukUPah0s5egY'
access_key= '1289080331111407617-TRaT1X3mL1CGADHCxP69aYmY8Feomz'
access_secret = '62s2gDSkeVPbMJb8cyazY2qGaif1A9jwa3XHL7bWzcPkI'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

csvFile = open('tweets', 'a')
csvWriter = csv.writer(csvFile)

search_words = input('enter your search word:  ') # enter your words
new_search = search_words + " -filter:retweets"

for tweet in tweepy.Cursor(api.search,q=new_search,count=100,lang="en",since_id=0).items(10):
    csvWriter.writerow([tweet.text.encode('utf-8')])