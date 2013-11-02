import LWFG, LWFGParser
if __name__ == '__main__':
    gram = LWFG.parse_lwfg('LWFG_TESTGRAM.txt', 'LWFG_TESTLEX.txt')
    parser = LWFGParser.LWFGParser(gram)

    randomSents = [l.strip().split(',') for l in open('randomSents.csv').readlines()]
    
    opf = open('randSentEval_lwfg.csv', 'w')
    opf.write('Idx,Sent,NWords,Time,numTrees,ChartSize,GoldSuccess\n')
    for idx, sentItem in enumerate(randomSents):
        sent,gold = sentItem
        time,chartSize,numTrees,foundGold = parser.parse(sent.split(), True, gold)
        opf.write(str(idx+1)+',')
        opf.write(sent.replace(' ','.')+',')
        opf.write(str(len(sent.split()))+',')
        opf.write(str(time)+',')
        opf.write(str(numTrees)+',')
        opf.write(str(chartSize)+',')
        opf.write(str(foundGold)+'\n')

        print idx+1, foundGold
##        trees = parser.parse(sent.split())

##        opf.write("*"*len(sent)+'\n')
##        opf.write(sent + " : \n")
##        opf.write("*"*len(sent)+'\n')
##        for tree in trees:
##            opf.write(str(tree)+'\n')
##            opf.write(str(tree.node.body())+'\n')
##            opf.write("*"*len(sent)+'\n')
    opf.close()

##        print "*"*len(sent)
##        print sent + " : "
##        print "*"*len(sent)
##        for tree in trees:
##            print tree
##            print tree.node.head()
##            print tree.node.body()
##            print tree.node.string()
##            print '*'*len(sent)
