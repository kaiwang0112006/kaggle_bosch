def makesubmit(clf,testdf,featurelist,output="submit.csv"):
    testdf = testdf.fillna(0)
    feature_test = testdf[featurelist]
    
    pred = clf.predict(feature_test)
    
    ids = list(testdf['Id'])
    
    fout = open(output,'w')
    fout.write("Id,Response\n")
    for i,id in enumerate(ids):
        fout.write('%s,%s\n' % (str(id),str(pred[i])))
    fout.close()