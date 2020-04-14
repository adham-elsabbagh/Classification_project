import pandas as pd

new_potential = pd.read_csv("newPotential.csv", index_col="ID")
high_income_count=0
low_income_count=0
file = open("ids.txt","w")
file.write('ID'+' '+'Class'+'\n')
for i,c in new_potential.iterrows():
    if c['class']=='>50K':
        high_income=c['class']
        file.write(str(i)+'  '+high_income+'\n')
        high_income_count+=1
    # else :
    #     low_income=c['class']
    #     # file.write(str(i)+'  '+low_income+'\n')
    #     low_income_count +=1
file.close()
print('high income',high_income_count)
print('low income',low_income_count)
expected_high_income=(high_income_count*10)/100
expected_low_income=(low_income_count*5)/100
profit_high=(expected_high_income*980)-(high_income_count*10)
# profit_low=(expected_low_income*310)-(low_income_count*10)
# total_profit=profit_high+profit_low
print('total profit for high income',profit_high)
# print('profit for low income',profit_low)
