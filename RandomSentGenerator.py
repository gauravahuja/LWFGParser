import nltk, random, time
import LWFG
import pickle
from nltk.featstruct import FeatStruct as FS

class LWFGRandomSentGenerator():
    def __init__(self, grammar):
        self.grammar = grammar

    def doesAlign(self, ntHead, pCons):
        pVars = pCons.variables()
        ntHead = ntHead.rename_variables(used_vars = pVars)

        ntHeadIdx = ntHead.keys()[0]
        pFeats = set(pCons[ntHeadIdx].keys())
        nthFeats = set(ntHead[ntHeadIdx].keys())

        return (pFeats <= nthFeats) and (pCons.unify(ntHead) != None)
        

    def randomExpansion(self, nt, idx, pCons):
        prods = self.grammar.productions(nt)
        if not prods: # something went wrong, dead-end
            return None

        if not pCons: # i.e. nt == start state
            return random.choice(prods)
        
        random.shuffle(prods)
        for prod in prods:
            head = FS()
            head['h'+str(idx)] = prod.phi_c()['h']
            if self.doesAlign(head, pCons):
                return prod
        else:
            return None

    def updateCons(self, cons, idx, symHead):
        head = FS()
        head['h'+str(idx)] = symHead
        consVars = cons.variables()
        head.rename_variables(used_vars = consVars)

        return cons.unify(head)

    def genRandom(self, nt, idx, pCons, rules, depth=0):
        if depth > 40:
            return [None, None, None]
        
        sentence = ''
        randExp = self.randomExpansion(nt, idx, pCons)
        if not randExp:
            return [None, None, None]
##        print randExp

        rexpCons = randExp.phi_c()
        rules.append(randExp)
        for i,sym in enumerate(randExp.rhs()):
            if LWFG.is_nonterminal(sym):
                phrase,symHead,rules = self.genRandom(sym, i+1, rexpCons, rules, depth+1)
                if not phrase: # encountered a dead-end, collapse
                    return [None,None,None]

                rexpCons = self.updateCons(rexpCons, i+1, symHead)
                if not rexpCons:
                    return [None,None,None]
                
                sentence += phrase
            else:
                sentence += (sym + ' ')
        return [sentence, rexpCons['h'], rules]

    def genSentence(self):
        sent = None
        while not sent:
            sent,head,rules = self.genRandom(self.grammar.start(), 0, None, [])
            if sent and len(sent.split()) > 10:
                sent = None
        return [sent, rules]

def dtHelper(rules, idx):
    if LWFG.is_terminal(rules[idx].rhs()[0]):
        node = '(' + str(rules[idx].lhs()) + ' ' + rules[idx].rhs()[0] + ')'
        return [node, idx]
    
    tree = '(' + str(rules[idx].lhs()) + ' '
    lev = idx
    for i in range(1,len(rules[idx].rhs())+1):
        node,lev = dtHelper(rules, lev+1)
        tree += node
    tree += ')'
    return [tree, lev]

def drawTree(rules):
    tree, lev = dtHelper(rules, 0)
    return tree
    

if __name__ == '__main__':
##    gram = LWFG.parse_lwfg('../MyLWFGGrammar.txt', '../LWFG_Lexicon.txt')
    gram = LWFG.parse_lwfg('LWFG_TESTGRAM.txt', 'LWFG_TESTLEX.txt')
    generator = LWFGRandomSentGenerator(gram)

    ruleCoverage = {}
    for prod in gram.productions():
        ruleCoverage[prod] = 0
        
##    sent, rules = generator.genSentence()
##    for rule in rules:
##        ruleCoverage[rule] += 1

    sentences = []
    trees = []
    for i in range(7500):
        sent, rules = generator.genSentence()
        sentences.append(sent)
        trees.append(drawTree(rules))
        for rule in rules:
            ruleCoverage[rule] += 1
        print i+1, sent

    f = open('RULE_COVERAGE_INFO.pkl', 'w')
    pickle.dump(ruleCoverage, f)
    f.close()

    opf = open('randomSents.csv', 'w')
    for idx, sent in enumerate(sentences):
        opf.write(sent + ',')
        opf.write(trees[idx] + '\n')
    opf.close()
