with open('g_27_random_outputs.txt', 'rb') as file:
    f = file.readlines() 

data = []

for ind, row in enumerate(f):                      
    data.append(row.split())

#print(data)

for row in data:
    print(row[2] + row[2])
    #row[2] = str(row[2]) + str(row[2])   
    for idx in range(3, len(row)):
	row[2] = row[2] + ' ' # how to do it???
    	row[2] = row[2] + row[idx]                                
    print(row[2])