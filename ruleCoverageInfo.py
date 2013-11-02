import pickle
import LWFG

if __name__ == '__main__':
    f = open('RULE_COVERAGE_INFO.pkl')
    ruleCoverage = pickle.load(f)
    f.close()

    ptcount = 0
    for rule in ruleCoverage:
        if LWFG.is_terminal(rule.rhs()[0]):
            ptcount += 1
    print len(ruleCoverage.keys()), ptcount

    count = 0
    lcount = 0
    for rule in ruleCoverage:
        if ruleCoverage[rule] == 0:
            count += 1
##            print rule
            if LWFG.is_terminal(rule.rhs()[0]):
                lcount += 1
            
    print count, lcount
    f = open('RULE_COVERAGE_INFO.csv', 'w')
    f.write('Rule,Count\n')
    for rule,count in ruleCoverage.items():
        f.write(str(rule)+','+str(count)+'\n')
    f.close()
