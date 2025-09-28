from re import sub#, compile
#from nltk.corpus import stopwords

#converts a probability distribution list to integer form and normalizes to a sum of 100
#if the sum of integer probabilities != to 100, adds or subtracts the excess/remaining value from the lowest probability
def normalize_to_int(probability_distribution):
    probability_distribution = [round(probability * 100) for probability in probability_distribution[0]]
    least_value = probability_distribution.index(min(probability_distribution))
    rem = 100 - sum(probability_distribution)
    probability_distribution[least_value] += rem

    return probability_distribution


#preprocesses the input to make it appropriate for the article classifier
def complete_preprocessing(article):
    article = str(article).lower() #converts all letters to lowercase
    article = sub('[^a-zA-Z]', ' ', article) #removes all non-alphabetical characters
    article = sub('\s+[^a-zA-Z]\s+', ' ', article) #substitutes all excess whitespace characters to one per

    #!!!TODO - not working, error thrown for re compilation
    #pattern = compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    #article = pattern.sub('', article)

    return article