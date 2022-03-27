
# =============================================================================
# Created on Wed Mar 16 16:13:43 2022
 
#  @author: Abhishek Nayak
# =============================================================================

# =============================================================================
#                          IMPORT LIBRARIES
# =============================================================================
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
#                             IMPORT DATASET
# =============================================================================

# jobs
url = 'https://raw.githubusercontent.com/abheshek-nayak/job-recommendation/main/job.csv'
dataset = pd.read_csv(url,dtype=str)
dataset2 = dataset[['UG_Degree','UG_Specialization','Interest','Skills', 'CGPA','Certification_name','Job_Title']]
dataset3 = dataset2.copy() # this is to prevent any errors that may arise due to modification

#Higher Education
pg = 'https://raw.githubusercontent.com/abheshek-nayak/job-recommendation/main/pgdata.csv'
dataset_pg = pd.read_csv(pg,dtype=str)
dataset_pg = dataset_pg[['UG','UG Stream','Interest','Skills', 'CGPA','PG']]
dataset_pg2 = dataset_pg.copy() # this is to prevent any errors that may arise due to modification

# =============================================================================
#                   CLEAN THE COLUMNS
# =============================================================================

# This is remove symbols,remove white and inbetween spaces and lowercasing the words.
def CLEAN(x):
    x = re.sub("[\'\"\-\/]+","", x)
    x = x.lower()
    x = x.strip()
    x = x.replace(';', ',')
    x = x.replace(" ", "")
    return x
    
#Cleaning work for job data
dataset3['UG_Degree'] = dataset3['UG_Degree'].apply(lambda x: CLEAN(x))
dataset3['UG_Specialization'] = dataset3['UG_Specialization'].apply(lambda x: CLEAN(x))
dataset3['Interest'] = dataset3['Interest'].apply(lambda x: CLEAN(x))
dataset3['Skills'] = dataset3['Skills'].apply(lambda x: CLEAN(x))
#dataset3['Certification_name'] = dataset3['Certification_name'].apply(lambda x: CLEAN(x))

#cleaning work for PG data
dataset_pg2.dropna(subset =(['UG Stream','Skills','PG']),inplace=True)

dataset_pg2['UG'] = dataset_pg2['UG'].apply(lambda x: CLEAN(x))
dataset_pg2['UG Stream'] = dataset_pg2['UG Stream'].apply(lambda x: CLEAN(x))
dataset_pg2['Interest'] = dataset_pg2['Interest'].apply(lambda x: CLEAN(x))
dataset_pg2['Skills'] = dataset_pg2['Skills'].apply(lambda x: CLEAN(x))



# =============================================================================
                          #BUILDING THE METADATA
# =============================================================================

#Metadata for jobs
features = ['UG_Degree','UG_Specialization','Interest','Skills']
dataset3['metadata'] = dataset3[features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
dataset3['metadata'] = dataset3['metadata'].str.replace(',', ' ')

#dataset3.iloc[0]['metadata']

#Metadata for Higher education
features_pg = ['UG','UG Stream','Interest','Skills']
dataset_pg2['metadata'] = dataset_pg2[features_pg].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
dataset_pg2['metadata'] = dataset_pg2['metadata'].str.replace(',', ' ')

# =============================================================================
#                              COUNTVECTOR FUNCTION
# =============================================================================
def get_vectors(*strs):
	text = [t for t in strs]

	vectorizer = CountVectorizer(text)
	vectorizer.fit(text)
	return vectorizer.transform(text).toarray()

# =============================================================================
#                         COSINE SMILIARITY FUNCTION
# =============================================================================
def get_cosine_sim(*strs):
	vectors = [t for t in get_vectors(*strs)]

	sim = cosine_similarity(vectors)
	return round(float(sim[0,1]),4)

# =============================================================================
#                   CREATE THE MULTI SELECT OPTIONS FOR USERS
# =============================================================================

# FOR CLEANING THE USER INPUT
def CLEAN2(X):
    #X = X.lower()
    X = X.replace(';', ',')
    X = X.strip()
    return X

# CREATING A MULTISELECTION OPTIONS
degree_scroll = dataset2["UG_Degree"].apply(lambda x: CLEAN2(x)).unique()
spec_scroll = dataset2['UG_Specialization'].apply(lambda x: CLEAN2(x)).unique()

dataset2['Interest'] = dataset2['Interest'].apply(lambda x: CLEAN2(x))
interest_scroll = (','.join(dataset2['Interest'])).split(',')
interest_scroll = list(set(interest_scroll))

dataset2['Skills'] = dataset2['Skills'].apply(lambda x: CLEAN2(x))
skills_scroll = (','.join(dataset2['Skills'])).split(',')
skills_scroll = list(set(skills_scroll))

# =============================================================================
                     #STREAMLIT INITIALIZTION
# =============================================================================
import streamlit as st
st.header("Career Recommendation For you")
degree = st.selectbox("Your Undergraduation Degree",
                             options=degree_scroll)
Specialization = st.selectbox("Your Specialization",
                             options=spec_scroll)
Interests = st.multiselect('Your interests',
                        options=interest_scroll)
Skills = st.multiselect('Your Skills',
                        options=skills_scroll)

submit = st.button("Recommend Job")
submit2 = st.button("Recommend Higher Education")


#BUILD A DICTIONARY FROM USER INPUT
data = {'Degree':degree,"Specialization":Specialization,"Interests":Interests,"Skills":Skills}

# join the dictionary values to form a string
test2 = ','.join(str(x) for x in data.values())

#cleaning of user inputs
test2 = test2.replace(' ', '')
test3 = test2.replace(',', ' ')
test3 = test3.lower()

#APPLYING COSINE SIMILARITY AND SORTING ACCORDINGLY.
dataset3['similarity']=0
dataset3['similarity'] = dataset3['metadata'].apply(lambda x: get_cosine_sim(x,test3))
dataset3 = dataset3.sort_values(by = 'similarity',ascending = False)


jobs = (','.join(dataset3['Job_Title'])).split(',')

jobs2 = pd.DataFrame(jobs)
jobs2.columns =['Jobs recommended for you']
jobs2 = jobs2.drop_duplicates()



#APPLYING COSINE SIMILARITY AND SORTING for PG DATA
dataset_pg2['similarity']=0
dataset_pg2['similarity'] = dataset_pg2['metadata'].apply(lambda x: get_cosine_sim(x,test3))
dataset_pg2 = dataset_pg2.sort_values(by = 'similarity',ascending = False)

education = (','.join(dataset_pg2['PG'])).split(',')

education2 = pd.DataFrame(education)
education2.columns =['Higher Education recommended for you']
education2 = education2.drop_duplicates()


if submit:
    st.table(jobs2[:5])
    # st.table(education2[:5])

if submit2:
    st.table(education2[:5])
