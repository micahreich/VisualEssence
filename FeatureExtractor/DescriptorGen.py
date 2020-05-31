import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re


class DescriptorGenerator:
    def create_stop_words(self, path_to_list):
        stopwords_list = []

        f = open(path_to_list, "r")
        for i in f:
            if len(i) != 0:
                stopwords_list.append(i.replace("\n", ""))

        return stopwords_list

    def get_descriptions(self, path_to_csv):
        f = open(path_to_csv)
        csv_f = csv.reader(f)

        desc_corpus = []

        i = 0
        for row in csv_f:
            if len(row) != 0 and len(row[27]) != 0 and len(row[28]) != 0 and len(row[29]) != 0:
                desc_corpus.append(row[27].strip())
                desc_corpus.append(row[28].strip())
                desc_corpus.append(row[29].strip())

                i += 1

        for j in range(len(desc_corpus)):
            desc_corpus[j] = desc_corpus[j].lower()
            regex = re.compile('[1234567890,./?]')
            desc_corpus[j] = regex.sub('', str(desc_corpus[j]))

        del desc_corpus[0:3]

        return desc_corpus

    def tf_idf(self, train_corpus, n_terms=1):
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, stop_words=self.create_stop_words(
            'description_data/stopwords.txt'))
        tfidf_vectorizer.fit_transform(train_corpus)

        tfidf_descriptors = []

        new_descriptors = []

        for i in range(0, len(train_corpus)):

            tfidf_desc = tfidf_vectorizer.transform([train_corpus[i]])
            feature_array = np.array(tfidf_vectorizer.get_feature_names())

            tfidf_sorting = np.argsort(tfidf_desc.toarray()).flatten()[::-1]

            top_n = feature_array[tfidf_sorting][:n_terms]

            new_descriptors.append(top_n[0])

            if (i+1) % 3 == 0:
                tfidf_descriptors.append(new_descriptors)
                new_descriptors = []

        return tfidf_descriptors


Descriptions = DescriptorGenerator()
data = Descriptions.get_descriptions("description_data/DESC450.csv")
print(np.array(Descriptions.tf_idf(data))[0])

