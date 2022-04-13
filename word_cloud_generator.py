from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
import matplotlib.pyplot as plt

temp = pd.read_excel('complexest_tokens_freq.xlsx', index_col=None, dtype=str, header=0,
                     sheet_name='complexest_tokens_freq', usecols="A:L")

print(temp)
temp["Positive Vs Negative Dif"] = temp["Positive Vs Negative Dif"].astype(float).mul(-1.0)
print(temp["Positive Vs Negative Dif"])
temp["Positive Vs Negative Dif"].mul(-1.0)
print(temp["Positive Vs Negative Dif"])
temp.sort_values(by=["Positive Vs Negative Dif"])

d = dict(zip(temp["Token"].astype(str)[-50:], temp["Positive Vs Negative Dif"].astype(float)[-50:]))

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(d)



plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()