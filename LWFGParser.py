#!/usr/bin/python

""" Parser for Lexicalized Well-Founded Grammars.

@author: gaurav kharkwal
"""

import nltk
import LWFG
import re
import time

from nltk.featstruct import FeatStruct as FS
from nltk.sem.logic import Variable
##
##class PerfStat:
##    def __init__(self, time, chartSize, numDerivs, parseSuccess):
##        self.parsingTime = time
##        self.chartSize = chartSize
##        self.numDerivs = numDerivs
##        self.success = parseSuccess
##    

class State:
    """
	Class representing a parsing ``state.''

	A state represents a production rule complete with parsing info, such as 
	position of the ``dot'' (see Chart Parsing guides for info on what that means),
	indices corresponding to the set of words that the production encompasses,
	the semantic head, semantic body, and the string corresponding to that production,
	and whether it has been completely satisfied.

	The state consists of the following attributes:
	    1) prod: The LexProduction rule defining the class
		1.1) Reminder that LexProduction is similar to a CFG Production, but
		     also contains a FeatStruct encoding compositional constraints.
	    2) dotIdx: The index of the position of the ``dot.''  When the dot is in
		       position 0, it implies the production has recently been predicted.
		       Every subsequent act of completion moves the dot forward one step.
	    3) startIdx: The start position of the state.  
	    4) endIdx: The end position of the state. Taken together, StartIdx and EndIdx
		       imply the production rule covers words indexed from StartIdx+1 to 
		       EndIdx.
	    5) _satisfied: Boolean denoting whether the production rule has been completely
			   satisfied.  That is, the dot has reached the end and the 
			   compositional constraints were successfully applied.
	    6) _head: FeatStruct representing the semantic head of the LHS of the production.
	    7) _body: OntoSeR construct representing the semantic body of the LHS of the prod.
	    8) _string: String representing the set of words covered by the production.
    """
    def __init__(self, prod, dot_index, start_index, end_index, satisfied=False):
        """
	    Inits by saving the production rule, the dot index, and the start and end indices.
	    To ensure copy-by-reference does not cause a cascading effect when changing attributes,
	    a new LexProduction object is instantiated from the ``prod'' param.

	    The input production's ``phi_c'' (compositional constraints) is stored as the state's
	    semantic head.
	    If the LHS of the production rule is a non-terminal, its ``string'' and ``body'' are
	    stored as the state's string and semantic body.

	    The LHS is a terminal in case of the special lexical rules of the form:
		$word$ -> .
	    In that case, the state's string and semantic body are initialized as empty objects.

	    :param satisfied: True when the state has been completely satisfied, False otherwise.
	"""
        self.prod = LWFG.LexProduction(prod._lhs, prod._rhs, prod._phi_c)
        self.dotIdx = dot_index
        self.startIdx = start_index
        self.endIdx = end_index

        self.head = prod.phi_c() # semantic head, represented as NLTK's FeatStruct
        self.body = None # semantic body, represented as LWFG's OntoSeR
        self.string = "" # set of covered words
        if LWFG.is_nonterminal(prod.lhs()):
            self.string = prod.lhs().string()
            self.body = prod.lhs().body()

        self._satisfied = satisfied

    def copySemInfo(self, other):
        """ Copies the semantic head, semantic body, and the string of 
	    other to self.
	"""
        if isinstance(other, self.__class__):
            self.head = other.head
            self.body = other.body
            self.string = other.string
        else:
            raise AttributeError
        
    def isCompleted(self):
        """ Returns whether the state has been ``completed'' -- i.e. the dot
	    has reached the end of the rule.
	"""
        return self.dotIdx >= len(self.prod.rhs())

    def isSatisfied(self):
        """ Returns whether the state has been satisfied -- i.e. the state
	    has been completed and the compositional constraints have been applied.
	"""
        return self._satisfied

    def nextTerm(self):
        """ Returns the term following the dot, and None if the dot has reached the end.
	"""
        if self.isCompleted():
            return None
        return self.prod.rhs()[self.dotIdx]
 
    def __str__(self):
        """ Pretty prints the state using the production rule, an appropriately placed dot, the start
	    and end indices.
	"""
        terms = [str(t) for t in self.prod.rhs()]
        terms.insert(self.dotIdx, u".")
        return "%-5s -> %-16s [%s-%s]" % (str(self.prod.lhs()), " ".join(terms), self.startIdx, self.endIdx)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """ Returns true if self and other belong to the same class, and have the same production rule,
	    dot Idx, and semantic head and body. """
        try:
            return (isinstance(other, self.__class__)\
		    and (self.prod == other.prod)\
		    and (self.dotIdx == other.dotIdx)\
                    and (self.startIdx == other.startIdx)\
                    and (self.endIdx == other.endIdx)\
		    and (self.body == other.body)\
                    and (self.head == other.head)
##                    and (self.invokedBy == other.invokedBy)
		    )
        except AttributeError:
            return False

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        """ Returns the has value of the state, computed using the production rule,
	    the dot index, and the semantic head & body. 
	"""
        return hash((self.prod, self.dotIdx, self.body, str(self.head)))

   
class LWFGParser(object):
    """ Parser for Lexicalized Well-Founded Grammars.

	Parsing is essentially a bottom-up chart parsing process, with the only 
	addition being the application of compositional constraints at the end of
	state completions.

	While a detailed description of bottom-up chart parsing is redundant, it is 
	still interesting to discuss some aspects of how it has been implemented.
	The parsing process uses a chart, which is essentially a list of lists.  The 
	chart is of size length(words)+1.  
	Each list in the table consists of an unordered	set of states (see State class above), 
	which represent partially processed production rules.
	
	Parsing is initiated by adding empty lexical rules of the form: $word$ -> \epsilon
	to the charts.  
	The lexical rule corresponding to the ith word is placed in the	i+1th column of the chart.  
	Subsequently, prediction, completion, and satisfaction rules are applied to every
	completed state to generate new states.  
	Unlike the LWFG parser described by Muresan (2006), this parser does not wait till the
	states are completed before applying the compositional constraints. 
	Instead, constraints are applied every time the dot position is moved and are used to ensure
	a satisfied state can be a constituent of another (i.e. moving the dot is okay given that
	satisfied state.)
	In doing so, we are able to reduce the number of states that are generated by pruning away
	states that are incompatible with the compositional constraints as soon as they become
	incompatible.

	Parsing is terminated when no new states are generated.
	Parsing is successful iff there is a state containing the grammar's start symbol in 
	its production rule's LHS, and the state covers all words (startIdx=0, endIdx=length(words)+1).

	Constituent information is stored at the end of every completion step which is later
	used to generate parse trees.

	The following attributes are used:
	    1) gram: the LWFG for the parser.
	    2) chart: a list of lists that stores states.
	    3) _chartOfSets: a list of sets for faster searching.
	    4) _constituentInfo: a dictionary that stores constituent info for every state.
			         The const. info comprises which states were used to complete
				 the key state.
	    
    """
    def __init__(self, grammar):
        """ Inits by saving the LWFG and initializing an empty chart and _chartOfSets.
	"""
        self.gram = grammar
        self.chart = []
        self._chartOfSets = []

    def _initializeChart(self, words):
        """ Initializes chart given a list of words comprising the sentence to be parsed.

	    Chart length is set to len(words)+1.
	    Empty lexical rules of the form: ``$word_i$ -> '' are created with empty 
	    compositional constraints, and added to the (i+1)th position in the chart.
	"""
        self._constituentInfo = {}
        self.chart = []
        self._chartOfSets = []

        for i in range(len(words)+1):
            self.chart.append([])
            self._chartOfSets.append(set())

        for i in range(len(words)):
            initRule = LWFG.LexProduction(words[i], [], None)
            self._addToChart(State(initRule, 0, i, i+1, True), i+1)

    def _chartLen(self):
        """ Returns the total number of states in the whole chart.
	"""
        return sum([len(col) for col in self.chart])

    def _addToChart(self, state, chartIdx):
        """ Adds the given state to the given position of the chart.
	    Returns True if state successfully added, False otherwise.

	    :param chartIdx: chart position at which state needs to be inserted
	"""
        if chartIdx >= 0 and chartIdx < len(self.chart):
            if state not in self._chartOfSets[chartIdx]:
                self.chart[chartIdx].append(state)
                self._chartOfSets[chartIdx].add(state)
                return True
        return False

    def _cleanChart(self):
        """ Removes all states from the chart that have not been satisfied.
	"""
        for i in range(len(self.chart)):
            self.chart[i][:] = [st for st in self.chart[i] if (st.isSatisfied())]

    def _printChart(self):
        """ Prints the chart in an easy to read fashion.
	"""
        for i in range(len(self.chart)):
            print '='*40
            print i, ':-'
            for st in self.chart[i]:
                print '   ', st
        print '='*40

    def _buildTrees(self, state):
        lhs = LWFG.LexNonterminal(state.prod.lhs())    
        lhs.setHead(state.head['h'])
        lhs.setBody(state.body)
        lhs.setString(state.string)
        
        if LWFG.is_terminal(state.prod.rhs()[0]):
            return [nltk.Tree(lhs, state.prod.rhs())]
        
        trees = []
        for constituent in self._constituentInfo[state]:
            for consTree in self._constituentTrees(constituent, 0, []):
                trees.append(nltk.Tree(lhs, consTree))
        return trees

    def _constituentTrees(self, constituent, childID, subTree):
        if childID == len(constituent):
            return [subTree]
        
        trees = []
        for cTree in self._buildTrees(constituent[childID]):
            for t in self._constituentTrees(constituent, childID+1, subTree+[cTree]):
                trees.append(t)
        return trees
            
            
##    def _buildTrees(self, state):
##        """ Recursively generates a parse tree that starts from the given state.
##
##	    Because LWFG produces unambiguous trees, only one tree needs to be generated.
##	    Tree generation proceeds simply by using ``self._consituentInfo'' to retrieve
##	    states that helped complete the curret state, and recursing through the process
##	    till pre-terminal rules are reached.
##
##	    --- THAT STATEMENT IS NOT TRUE ---
##
##	    The tree is represented as an NLTK Tree.
##	    Only non-terminals need to be saved (barring the leaves), and are freshly 
##	    instantiated with their semantic head, semantic body, and string being 
##	    retrieved from the corresponding states.
##
##	    :rtype nltk.Tree
##	"""
##        if LWFG.is_terminal(state.prod.rhs()[0]):
##            lhs = LWFG.LexNonterminal(state.prod.lhs())
##            
##            lhs.setHead(state.head['h'])
##            lhs.setBody(state.body)
##            lhs.setString(state.string)
##            return nltk.Tree(lhs, state.prod.rhs())
##
##        lhs = LWFG.LexNonterminal(state.prod.lhs())            
##        lhs.setHead(state.head['h'])
##        lhs.setBody(state.body)
##        lhs.setString(state.string)
##        trees = []
##        for constituent in self._constituentInfo[state]:
##            children = []
##            for child in constituent:
##                children.append(self._buildTrees(child))
##            for subtree in self._cartProduct(children)
##                
##        
##        children = []
##        for child in self._constituentInfo[state][0]:
##            children.append(self._buildTrees(child))
##
##        lhs = LWFG.LexNonterminal(state.prod.lhs())
##            
##        lhs.setHead(state.head['h'])
##        lhs.setBody(state.body)
##        lhs.setString(state.string)
##
##        return nltk.Tree(lhs, children)
 

    def _predictor(self, state):
        """ Predictive step: 
		Given a satisfied state, new states are generated such that their production
		rules contain the LHS of the satisfied state as the first ter.
		The dot position of the new states are initialized at the front of the RHS,
		and the start and end indices initialized to the start index of the satisfied state.
	"""
        start = state.startIdx

        for prod in self.gram.productions(rhs = state.prod.lhs()):
            predicted = State(prod, 0, start, start)
            if predicted not in self._chartOfSets[start]:
##                predicted.invokedBy = str(state)
                self._addToChart(predicted, start)

    def _completer(self, state):
        """ Completion step:
		Given a satisfied state, the completion process moves the dots of other existing 
		states in the following fashion.
		For every state such that the next term following the dot index is the same
		as the LHS of the satisfied state, the dot is moved forward by one step.
		A new state is generated with the dot moved by 1, and an updated end index, which is 
		now equal to the end index of the satisfied state.

		The semantic information from the previous state (which was modified to generate the
		new state) is copied to the new state.

		Lastly, compositional constraints are applied, and if the process is successful, the 
		new state is added to the chart at the end index of the satisfied state. 
		The constituent information corresponding to the new state is updated to reflect
		the role of the satisfied state.
	"""
        stStart = state.startIdx
        stEnd   = state.endIdx
        stLHS   = state.prod.lhs()
        
        for aState in self.chart[stStart]:
            if aState.nextTerm() == stLHS:  # the state meets the requirement of having its next term be 
					    # equal to the LHS of the satisfied state
                newState = State(aState.prod, aState.dotIdx+1, aState.startIdx, stEnd)

                if LWFG.is_nonterminal(aState.prod.lhs()): # making sure it's not a ``lexical'' state
                    newState.copySemInfo(aState)

                if newState not in self._chartOfSets[stEnd]:
                    #print 'applying constraints... ',
                    #t = time.time()
                    status, newState = self._applyConstraints(newState, state)
                    #print 'done', time.time()-t
                    if status == True: # newState and state are compatible
                        if newState.isCompleted():
                            newState._satisfied = True
##                        if str(newState.prod) == "sbj -> n":
##                            print newState, state
##                            print repr(newState.head['h'])
##                            print newState.body

##                        newState.invokedBy = str(state)
                        self._addToChart(newState, stEnd)

                        
                        if str(newState.prod) == "nc -> noun":
                            print state, state.body, repr(state.head)
                            print newState

			# now updating constituent information
                        constituents = []
                        for cons in self._constituentInfo.get(aState,[]):
                            constituents.append(cons+[state])
                        if len(constituents) == 0:
                            constituents.append([state])

                        if newState not in self._constituentInfo.keys():
                            self._constituentInfo[newState] = []

                        for cons in constituents:
                            if cons not in self._constituentInfo[newState]:
                                self._constituentInfo[newState].append(cons)
                        
    def _newVarName(self, var):
        """ Helper function that renames a given variable name.
        """
        if var[-1].isdigit():
            return var[:-1]+str(int(var[-1])+1)
        else:
            return var+"2"

    def _applyConstraints(self, parent, child):
        """ Constraint step:
		Given a satisfied state ``child'' and the updated state ``parent,'' compositional constraints
		are applied in the form of unification between the semantic heads of the two states.
		A successful unification implies the two states are compatible given the compositional 
		constraints.
		In which case, the semantic bodies are merged and updated to reflect the unification.
		Lastly, the string of the ``parent'' is updated.

		Because unification is a tricky process, care needs to be taken to ensure that it is properly
		performed.
		Variables that have the same name are assumed to be the same, but because the grammar generates
		feature structures for compositional constraints using a small set of variable names, often
		variables can have the same name despite their being independent entities.
		One workaround is to rename the variables in one of the semantic heads before unification.
		It is important that the renamed variables are updated in the semantic bodies as well.
	"""
        if LWFG.is_terminal(child.prod.lhs()):	# child is a special lexical state, and thus contains no semantic info
            return [True, parent]		# skip

	###################################################
	# Unification of Semantic Heads
	##################################################

	# The compositional constraints are represented in such a way that there a separate set of constraints for each
	# term in the production rule.
	# Each set of constraints is a separate feature structure.
	# The set of constraints for the LHS are indexed with a feature identifier: `h'
	# The set of constraints for the ith term in the RHS are indexed with a feature indentifier: `hi'
	#
	# Thus, when unification is performed, the constraints of the LHS of the child state (indexed by `h') need
	# to be retrieved.
	# Also, because NLTK unifies embedded feature structures only when their feature identifiers are the same,
	# we need to have the feature id of the retrieved set of constraints to match the corresponding feature id
	# in the parent state.
	# Again, that would be `hi' where $i$ is the index of the child's LHS in the parent's RHS.
        hidx = 'h'+str(parent.dotIdx)
        childHead = FS()  # make a new Feature Structure to get around the copy-by-ref issues
        childHead[hidx] = child.head['h']
    
        parentHead = parent.head
        
        # Rename variables in childHead to avoid confusion
        # step 0: check if child.body and parent.body share variable names, change any that are shared
        renamedVarMap1 = {}
        usedVars = []
        if parent.body:
            pBodVars = parent.body.variables()
            cBodVars = child.body.variables()
            for v in cBodVars:
                if v in pBodVars: # change v
                    nv = self._newVarName(v)
                    while nv in pBodVars+cBodVars:
                        nv = self._newVarName(nv)
                    usedVars.append(Variable(nv))
                    renamedVarMap1[Variable(v)] = Variable(nv)
                else: # not shared by parent
                    usedVars.append(Variable(v))
            for v in pBodVars:
                usedVars.append(Variable(v))
        else:
            for v in child.body.variables():
                usedVars.append(Variable(v))
                    

        # step 1: find used variables from parent's semantic head  
        usedVars += list(parentHead.variables())

        # step 2: rename variables in child's semantic head
        renamedVarMap2 = {}
        childHead = childHead.rename_variables(used_vars=usedVars, new_vars=renamedVarMap2)

	# check if features align with each other
        childFeats = set(childHead[hidx].keys())
        parentFeats = set(parentHead[hidx].keys())
        if not (parentFeats <= childFeats): # True if the relevant set of features of the parent state
					    # are **not** a subset of those of the child state.
            return [False, parent]	    # If True, the states are incompatible.
        
	# perform unification
        bindings = {}
        parentHead = parentHead.unify(childHead, bindings)
        if not parentHead: # failed to unify
            return [False, parent] # the states are incompatible

        # update pState's rule
        parent.head = parentHead

	###################################################
	# Updating Semantic Bodies
	##################################################
        childBody = LWFG.OntoSeR(str(child.body)) # create new sem body to get around copy-by-ref issues

        # The next part is a bit confusing and needs spelling out.
        # We want to change variables names in the child's semantic body that are shared with
        # the parent's semantic body.
        # However, we can't just do it willy-nilly, we need to ensure that only those variable names
        # that are not linked with the parent's body are changed.
        # For example, in "the smart girl" -- when "n -> det n" is completed, the body corresponding to "the"
        # is linked with the body corresponding to "smart girl".
        # If we change variables names without consideration, that link is lost.
        #
        # To maintain the link, a roundabout method is employed.
        # We first look at all renamed var names in "renamedVarMap2" -- which results from changing vars in
        # the child's semantic head.
        # We then look at all renamed names and see whether they were involved in unification with the parent's
        # head.
        # We can do that by going through the unification "bindings" and checking to see if the renamed names
        # are keys.
        # If they were involved in unification, we check to see if they were bound to a variable in the parent's
        # head that we were about to rename.
        # We can do that by looking to see if the bound variable is a key in "renamedVarMap1" -- which we got
        # from renaming variable names in the child's body that were present in the parent's body.

        for var in renamedVarMap2.keys():
            if renamedVarMap2[var] in bindings.keys() and bindings[renamedVarMap2[var]] in renamedVarMap1.keys():
                renamedVarMap1.pop(bindings[renamedVarMap2[var]])

        childBody.substituteBindings(renamedVarMap2)            
        childBody.substituteBindings(renamedVarMap1)

        if not parent.body:
##            print parent, child
##            print parent.body, child.body
##            print renamedVarMap1
##            print renamedVarMap2
##            print bindings
##            print " "
            parentBody = childBody
        else:
            
            parentBody = LWFG.OntoSeR(str(parent.body) + ',' + str(childBody))
        parentBody.substituteBindings(bindings)
        parent.body = parentBody

	###################################################
	# Updating Strings
	##################################################
        if not parent.string:
            parentString = child.string
        else:
            parentString = parent.string + ' ' + child.string
        parent.string = parentString
        
        return [True, parent]
           

    def parse(self, words, evaluate=False, gold=None):
        """
	    Actual parsing process.

	    The first step is to ensure that the input list of words is compatible with 
	    the grammar of the parser.
	    Subsequently, the chart is initialized with the list of words.
	    Parsing is a simple loop that goes through all states in the chart and finds
	    states that have been satisfied.
	    On detecting a satisfied state, the completer and predictor methods are called
	    that generate new states.
	    If no new states are generated, the parsing loop is terminated.

	    Lastly, the last column of the chart is searched through to find if there is any
	    state that contains the grammar's start symbol in its production rule's LHS and that
	    has a start index of 0 (implying it covers all words).
	    If there is such a state, parsing is said to be successful, otherwise a ValueError 
	    is raised.
	    For all such full coverage start states, parse trees are generated and returned 
	    as a list.

	    :param words: A list of strings that represents words in a sentence.
	    
	    :return: A list of parse trees, if parsing is successful
	    :rtype: list(nltk.tree.Tree)
	"""
        self.gram.check_coverage(words) # raises ValueError if incompatible
        
        startTime = time.time()
        
        self._initializeChart(words)

        modified = True
        while modified:
            modified = False
            oldLen = self._chartLen()
            for col in self.chart:
                for state in col:
#                    if state.isCompleted() and state.isSatisfied():
                    if state.isSatisfied():
                        self._completer(state)
                        self._predictor(state)

            newLen = self._chartLen()
            if newLen != oldLen:
                modified = True

##        self._printChart()
##        for k,v in self._constituentInfo.items():
##            print k,':',v
        

        success = False
        foundGold = False
        trees = []
        treeCount = 0
        for state in self.chart[-1]:
            if state.isSatisfied() \
               and state.prod.lhs() == self.gram.start() \
               and state.startIdx == 0:
                success = True
                for tree in self._buildTrees(state):
                    treeCount += 1
                    trees.append(tree)
                    if not foundGold:
                        tree = re.sub('\s+', ' ', str(tree))
                        tree = re.sub('\) \(', ')(', tree)
##                    print gold, ' == ', tree
                        if gold and tree == gold:
                            foundGold = True
        endTime = time.time()

        if not success:
            self._printChart()
            raise ValueError('Unable to parse sentence!')

        if evaluate == True:
            return [endTime-startTime, self._chartLen(), treeCount, foundGold]
        else:
            return trees
                    
if __name__ == '__main__':
    gram = LWFG.parse_lwfg('LWFG_TESTGRAM.txt', 'LWFG_TESTLEX.txt')
    parser = LWFGParser(gram)

    #sentence = "the girl on the table loved john"
    #sentence = "the smart girl was loved by john"
    #sentence = "the smart elephant on the table was loved by the girl from Spain"
    #sentence = "the smart girl from Spain loved the smart boy from Rome"
    #sentence = "the girl is walking on the table" # doesnt work
    sentence = "the girl loved the boy"
    #sentence = "disruptive treatments liked on how walked"

    #print parser.parse(sentence.split(),True)

    trees = parser.parse(sentence.split())

    print "*"*len(sentence)
    print sentence + " : "
    print "*"*len(sentence)
    for tree in trees:
        print tree
    
        print tree.node.head()
        print tree.node.body()
        print tree.node.string()
        print '*'*len(sentence)
