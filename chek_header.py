        
if not sniffer.has_header(open(outputfile).read(sample_bytes)): 
        with open(outputfile,'w') as writeFile:
            fieldnames = ['ID','Image','Sequence','Temperature','Time','Make',
              'Moonphase','ExposureTime','ISOSpeedRatings','SceneCaptureType',
              'WhiteBalance','ExposureMode','ImageLength','ImageWidth',
              'ColorSpace','FlashPixVersion','XResolution','YResolution',
              'ResolutionUnit','YCbCrPositioning','ExifOffset','ComponentsConfiguration',
              'Flash','FlashPixVersion','Saturation','Contrast','Sharpness',
              'Detection','Score','Detection_link']
            writer = csv.DictWriter(writeFile, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
        writeFile.close()
