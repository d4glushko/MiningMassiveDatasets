#%%
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, SparkContext, StopWordsRemover
from pyspark.sql import SparkSession
import numpy as np
from sklearn.neighbors import LSHForest

#%%
sc = SparkContext("local", "first app")

#%%
spark = SparkSession(sc)

#%%
df = spark.read.csv(
    "data/listings.csv", header=True, mode="DROPMALFORMED"
)
names_df = df.select('name')
names_df.show()
# listings.filter(listings["name"].isNotNull())

#%%
names_df = names_df.dropna(subset='name')
names_df.show()

#%%
tokenizer = Tokenizer(inputCol="name", outputCol="words")
wordsData = tokenizer.transform(names_df)
wordsData.show()

#%%
stopwords = []
stopwords.extend(StopWordsRemover.loadDefaultStopWords('english'))
remover = StopWordsRemover(inputCol="words", outputCol="cleanedWords", stopWords=stopwords)
cleanedWordsData = remover.transform(wordsData)
cleanedWordsData.show()

#%%
hashingTF = HashingTF(numFeatures=4096, inputCol="cleanedWords", outputCol="tfFeatures")
tfWordsData = hashingTF.transform(cleanedWordsData)
tfWordsData.show()

#%%
idf = IDF(inputCol="tfFeatures", outputCol="tfIdfFeatures")
idfModel = idf.fit(tfWordsData)
results = idfModel.transform(tfWordsData)
results.show()


#%%
python_df = results.toPandas()
python_df

#%%
series = python_df['tfIdfFeatures'].apply(lambda x : np.array(x.toArray())).as_matrix().reshape(-1,1)
features = np.apply_along_axis(lambda x : x[0], 1, series)
features

#%%
lshf = LSHForest(random_state=4)
lshf.fit(features)

#%%
queryPoints = [features[1], features[2]]

#%%
distances, indices = lshf.kneighbors(queryPoints, n_neighbors=2)
print(distances)
print(indices)
