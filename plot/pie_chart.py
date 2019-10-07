from matplotlib import pyplot as plt

payment_method_names = ["Card Swipe", "Cash", "Apple Pay", "Other"]
payment_method_freqs = [270, 77, 32, 11]

plt.pie(payment_method_freqs, autopct='%0.1f%%')
plt.axis('equal')
plt.legend(payment_method_names)
plt.show()
#%%
from matplotlib import pyplot as plt

unit_topics = ['Limits', 'Derivatives', 'Integrals', 'Diff Eq', 'Applications']
num_hardest_reported = [1, 3, 10, 15, 1]

plt.figure(figsize=(10,8))
plt.pie(num_hardest_reported, labels= unit_topics, autopct='%1d%%')
plt.axis('equal')
plt.title("Hardest Topics")
plt.savefig("my_pie_chart.png")
plt.show()

#%%
countries = [447, 66, 64, 49, 45, 28, 25, 20, 19, 17, 17, 279]
colors = ['lightskyblue', 'gold', 'lightcoral', 'gainsboro', 'royalblue', 'lightpink', 'darkseagreen', 'sienna', 'khaki', 'gold', 'violet', 'yellowgreen']
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
plt.pie(countries,explode = explode,colors=colors,
       shadow=True, startangle=345, autopct='%1.0f%%',
       pctdistance=1.15)
plt.title("League of Legends Viewers' Whereabouts")
plt.legend(labels, loc="right")
plt.show()
