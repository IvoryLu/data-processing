            with open('E:/test_wed/test_result.csv','r') as csvinput:             
                    reader = csv.reader(csvinput)
                    total = []
                    
#                    row.append('Animal_Detection')
#                    row.append(score)
#                    all.append(row)
                    
                    for row in reader:
                        
                        if row[1] == inputFileNames[iImage]:
                            row.append('animal')
                            row.append(score)
                        total.append(row)
            csvinput.close()
            
            with open('E:/test_wed/test_result.csv','w') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')            
                writer.writerows(total)    
            csvoutput.close()
