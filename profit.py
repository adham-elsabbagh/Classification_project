import pandas as pd
import classifier
new_potential = pd.read_csv("newPotential.csv", index_col="ID")
high_income_count=0
accuracy=classifier.results.mean()
low_income_count=0
file = open("ids.txt","w")
# file.write('ID'+' '+'\n')
for i,c in new_potential.iterrows():
    if c['class']=='>50K':
        high_income=c['class']
        file.write(str(i)+'\n')
        high_income_count+=1
file.close()

print('high income',high_income_count)
package_sent=high_income_count*10
profit=(high_income_count*accuracy*0.1*980)
cost=(high_income_count*(1-accuracy)*0.05*310)
total_profit=(profit-cost-package_sent)
print('Q1:',profit-package_sent)
print(' profit :',profit)
print(' cost :',cost)
print('total profit :',total_profit)
