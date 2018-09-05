from pandas import DataFrame


best_result = {'RA': [1,2,3], 'BA': [6,5,4], 'GA': [8,7,9], 'CMA': [18,11,2], 'PYS': [3,14,15]}
aaa = ['a', 'b', 'c']
vv = DataFrame(best_result, index=aaa)
print(vv)
for index, row in vv.iterrows():
    print(index)
    ddd = row.sort_values(ascending=False)
    print(ddd)
    print(ddd.values)