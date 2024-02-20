import nltk_spacy
import os
import textract
from tabulate import tabulate

resume_text = str(textract.process('data/resume/resume.pdf'), 'UTF-8').lower()
keywords_resume_spacy = nltk_spacy.spacy_keywords(resume_text)
keywords_resume_nltk = nltk_spacy.nltk_keywords(resume_text)

print('=' * 50)
print('RESUME KEYWORDS')
print('=' * 50)
print(f'{len(keywords_resume_spacy)} SpaCy keywords: {keywords_resume_spacy}')
print(f'{len(keywords_resume_nltk)} NLTK keywords: {keywords_resume_nltk}')
print('=' * 50)
print()

# keywords_resume_spacy = keywords_resume_nltk = ['python', 'java', 'c++', 'django', 'flask', 'sql', 'html', 'css', 'javascript', 'git', 'github', 'agile', 'postgresql', 'pandas']


# ---------------------------

# Getting job descriptions from all files in the directory
job_descriptions = []
for file_name in sorted(os.listdir('data/jd/')):
    jd_file_name = 'data/jd/' + str(file_name)
    job_descriptions.append({"filename": str(file_name), "text": str(textract.process(jd_file_name), 'UTF-8').lower()})

# ---------------------------

similarities = []

# Processing job descriptions
for job_description in job_descriptions:
    # Matching by Cosine Similarity
    text = [resume_text, job_description['text']]

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text)
    cosine = cosine_similarity(count_matrix)[0][1]

    # Matching with SpaCy keywords
    keywords_jd = nltk_spacy.spacy_keywords(job_description['text'])
    keywords_matched_spacy = set(keywords_jd).intersection(keywords_resume_spacy)
    percentage_spacy = len(keywords_matched_spacy) / len(keywords_resume_spacy)

    # Matching with NLTK keywords
    keywords_jd = nltk_spacy.nltk_keywords(job_description['text'])
    keywords_matched_nltk = set(keywords_jd).intersection(keywords_resume_nltk)
    percentage_nltk = len(keywords_matched_nltk) / len(keywords_resume_nltk)

    similarities.append([
        job_description['filename'],
        f'{cosine:.2%}',
        f'{percentage_spacy:.2%}',
        # ' '.join(keywords_matched_spacy),
        f'{percentage_nltk:.2%}',
        # ' '.join(keywords_matched_nltk)
    ])

print(tabulate(similarities, headers=[
    'File',
    'Cosine %',
    'SpaCy %',
    # 'SpaCy Keywords',
    'NLTK %',
    # 'NLTK Keywords'
]))
