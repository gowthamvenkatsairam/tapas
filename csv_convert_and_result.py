import csv

file = []
with open("balancesheet.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        file.append(row)
with open('questions.txt') as querylist:
    queryarray = querylist.read().splitlines()

result=predict(file,queryarray)
print(result)