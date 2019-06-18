#Simple one line header

with open('E:/test_wed/test_result.csv','w') as writeFile:
    fieldnames = ['ID','Image','Sequence','Detection','Score']
    writer = csv.DictWriter(writeFile, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()
writeFile.close()
