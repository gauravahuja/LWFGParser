import nltk, time, re, pickle

def getBaseGrammar(gramFname):
    gram = []
    fp = open(gramFname)
    for line in fp.readlines():
        if line[0] == '#':
            continue
        elif line[0] != '\t' and line.strip():
            gram.append(line.strip())
    return gram

def getBaseLexicon(lexFname):
    lex = []
    fp = open(lexFname)
    for line in fp.readlines():
        if line[0] == '#':
            continue
        elif line[0] != '\t' and line.strip():
            lex.append(line.strip())
    return lex

if __name__ == '__main__':
    gramLines = getBaseGrammar('LWFG_TESTGRAM.txt')
    lexLines = getBaseLexicon('LWFG_TESTLEX.txt')

    baseGram = nltk.grammar.parse_cfg('\n'.join(gramLines+lexLines))
    baseParser = nltk.parse.chart.BottomUpChartParser(baseGram)
    
    randomSents = [l.strip().split(',') for l in open('randomSents.csv').readlines()]
    opf = open('randSentEval_base.csv', 'w')
    opf.write('Idx,Sent,NWords,Time,numTrees,GoldSuccess\n')
    for idx, sentItem in enumerate(randomSents):
        sent,gold = sentItem
        opf.write(str(idx+1)+',')
        opf.write(sent.replace(' ','.')+',')
        opf.write(str(len(sent.split())) +',')
        t = time.time()
        trees = baseParser.nbest_parse(sent.split())
        foundGold = False
        for tree in trees:
            tree = re.sub('\s+', ' ', str(tree))
            tree = re.sub('\) \(', ')(', tree)
            if gold and tree == gold:
                foundGold = True
        t = time.time() - t
        opf.write(str(t) + ',')
        opf.write(str(len(trees))+',')
        opf.write(str(foundGold)+'\n')
        print idx+1,foundGold
    opf.close()

