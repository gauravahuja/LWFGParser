""" Class module for a lexicalized well-founded grammar (LWFG) (Muresan, 2010).

@author: gaurav kharkwal
@date: 20130730
"""

import re
import nltk
from nltk.featstruct import FeatStruct as FS
from nltk.sem.logic import Variable

class OntoSeR(object):
    """ Class representing OntoSeR (Ontology-based Semantic Representation, see Muresan (2006)).

        OntoSeR is of the following form:
            OntoSeR := AP | OntoSeR <lop> OntoSeR  // <lop> -- logical operator; currently not implemented
            AP      := ConceptID.Attr = Concept
            Concept := ConceptID | ConceptName
            Attr    := AttrID | AttrName

        Currently, the logical form is represented simply as a string.  In addition, instead of representing
        a single OntoSeR rule, the class can represent a set of OntoSeR rules separated by a comma.
        Such a representation facilitates use of the class to represent semantic bodies in LWFGs.
        
        To maintain consistency with the nltk's FeatStruct, variables (*IDs) are denoted by an initial `?'.
        Attribute names differ in that they do not contain a `?' at the beginning, and concept names are
        inclosed in quotes.
    """
    def __init__(self, ontosers):
        """ Inits by saving the input, and identifies the conceptIDs and attributeIDs. """
        self._ontosers = ontosers
        self._identifyVariables()

    def _identifyVariables(self):
        """ Identifies ConceptIDs and AttrIDs in the ontosers representation. """
        self._conceptIDs = set()
        self._attrIDs = set()

        for ontoser in self._ontosers.split(','):
            try:
                lhs,rhs = ontoser.split('=')
                if '.' in lhs:
                    cid,attr = lhs.split('.')
                    self._conceptIDs.add(cid)
                    if attr.startswith('?'):
                        self._attrIDs.add(attr)
            except ValueError:
                print ontoser
                raise ValueError
        
    def variables(self):
        """ Returns the list of ``variables'' in the OntoSeR.  Variables are defined to be
            ConceptIDs and AttrIDs.
        """
        return list(self._conceptIDs)+list(self._attrIDs)

    def _subHelper(self, ontoser, bindings):
        """ Helper function for self.substituteBindings.
            Takes the OntoSeR apart, and checks to see if any variable (*ID) is in the
            ``bindings'' dictionary.  Because the ``bindings'' dictionary contains NLTK's
            logic variables as keys, the appropriate transformation is first applied.
            If the variable is in ``bindings,'' it gets replaced.
        """
        s = r""
        lhs,rhs = ontoser.split('=')

        if '.' in lhs:
            lhs = lhs.split('.')
            s += '.'.join([str(bindings.get(Variable(l),l)) for l in lhs])
        else:
            s += str(bindings.get(Variable(lhs),lhs))
        s += "="
        s += str(bindings.get(Variable(rhs),rhs))
        return s

    def substituteBindings(self, bindings):
        """ Modifies the ontosers by replacing variables with substitution values present in the
            ``bindings'' dictionary.  Calls self._subHelper to help with the substitution process,
            and eventually recomputes the set of variables present in ``self.ontosers''.
        """
        if bindings:
            self._ontosers = ','.join([self._subHelper(term, bindings) for term in self._ontosers.split(',')])
            self._identifyVariables()

    def __eq__(self, other):
        """ Returns true iff ``self'' and ``other'' belong to the same class
            and have the same ontosers. """
        return (isinstance(other, self.__class__)\
                and (self._ontosers == other._ontosers) )

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self._ontosers)

    def __str__(self):
        return self._ontosers

    def __repr__(self):
        return str(self)
        
    

class LexNonterminal(object):
    """ Lexicalized non-terminal class.

        A nonterminal class that contains additional information needed for an LWFG. \
        Namely:
            1) Head: A Feature Structure containing compositional information.
            2) Body: Semantic representation ideally in a logical form but currently \
                     represented as a string
            3) String: The natural language string associated with the non-terminal.\
                        An empty string denotes an uninstantiated non-terminal.
                        
    """
    def __init__(self, symbol):
        """ Inits with empty head, body, and string. """
        self._symbol = symbol

        self._head = None
        self._body = None
        self._string = None
        
        self._hash = hash(symbol)

    def symbol(self):
        """ Returns the symbol of the non-terminal. """
        return self._symbol

    def head(self):
        """ Returns the head of the non-terminal. """
        return self._head

    def setHead(self, head):
        """ Sets the head of the non-terminal. """
        self._head = head

    def body(self):
        """ Returns the body of the non-terminal. """
        return self._body

    def setBody(self, body):
        """ Sets the body of the non-terminal. """
        self._body = body

    def string(self):
        """ Returns the string of the non-terminal. """
        return self._string

    def setString(self, string):
        """ Sets the string of the non-terminal. """
        self._string = string

    def __eq__(self, other):
        """ Returns true if ``self'' and ``other'' are both LexNonterminals \
            and share the same symbol. """
        try:
            return (isinstance(other, self.__class__)\
                    and (self._symbol == other._symbol) )
        except AttributeError:
            return False

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        """ Returns the hash value computed from the symbol during init. """
        return self._hash

    def __str__(self):
        return str(self._symbol)

    def __repr__(self):
        return str(self)

#--------------------------------------------#

def is_nonterminal(item):
    """ Returns whether the input item is a member of the LexNonterminal class. """
    return isinstance(item, LexNonterminal)

def is_terminal(item):
    """ Returns whether the input item is not a member of the LexNonterminal class.
        Essentially, an inverse of the is_nonterminal() method. """
    return hasattr(item, '__hash__') and not isinstance(item, LexNonterminal)

#--------------------------------------------#

class LexProduction(object):
    """ Lexicalized production class.

        Similar to the NLTK Production class in all but one aspect.
        Unlike the NLTK class, LexProduction contains an additional 
        param ``phi_c'' which represents the compositional constraints
        corresponding to the production.
        It is represented as an NLTK FeatStruct object.
    """
    def __init__(self, lhs, rhs, phi_c):
        """ Ensures the ``rhs'' param is a list, and if so instantiates by
            storing the ``lhs'', ``rhs'', and ``phi_c'' params.

            Simultaneously computes the hash value for the object.
        """
        if isinstance(rhs, (str, unicode)):
            raise TypeError('production right hand side should be a list, '
                            'not a string')
        self._lhs = lhs
        self._rhs = tuple(rhs)

        self._phi_c = phi_c
        
        self._hash = hash((self._lhs, self._rhs))

    def lhs(self):
        """ Returns the lhs of the production. """
        return self._lhs

    def rhs(self):
        """ Returns the rhs of the production. """
        return self._rhs

    def phi_c(self):
        """ Returns the compositional constraints (``phi_c'') of the production. """
        return self._phi_c

    def __len__(self):
        """ Overrides object.__len__().  Returns the length of the rhs. """
        return len(self._rhs)

    def __str__(self):
        s = '{!r} ->'.format(self._lhs)
        for trm in self._rhs:
            s += ' {!r}'.format(trm,)
        return s

    def __repr__(self):
        return '{!s}'.format(self)

    def __eq__(self, other):
        """ Returns true iff ``self'' and ``other'' belong to the same class
            and have the same lhs and rhs. """
        return (isinstance(other, self.__class__)\
                and (self._lhs == other._lhs) \
                and (self._rhs == other._rhs) \
                and (self._phi_c == other._phi_c) )

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        """ Returns the hash value computed during init. """
        return self._hash

#--------------------------------------------#

class LWFG(object):
    """ Lexicalized Well-Founded Grammar.

        The grammar consists of a start state and a set of lexicalized productions.
    """
    def __init__(self, start, productions):
        """ Creates a new LWFG from the given start state and a set of productions.

            Simultaneously generates a set of nonterminals and preterminals,
            and indices for faster retrieval of productions.
        """
        self._start = start
        self._productions = productions
        self._categories = set(prod._lhs for prod in productions)
        self._preterminals = set(prod._lhs for prod in productions if is_terminal(prod._rhs[0]))

        self._calculate_indexes()
        
    def _calculate_indexes(self):
        """ Generates a set of indices for fast search/retrieval.

            Creates 4 dictionaries:
                1) ``_lhs_index'': A list of productions indexed by the lhs.
                2) ``_rhs_index'': A list of productions indexed by the first entity in the rhs.
                3) ``_empty_index'': A dictionary of nonterminals that have empty productions.
                4) ``_lexical_index'': A list of productions indexed by the lexical tokens contained in the rhs.
        """
        self._lhs_index = {}
        self._rhs_index = {}
        self._empty_index = {}
        self._lexical_index = {}

        for prod in self._productions:
            lhs = prod._lhs
            self._lhs_index.setdefault(lhs, []).append(prod)
            if prod._rhs:
                rhs0 = prod._rhs[0]
                self._rhs_index.setdefault(rhs0, []).append(prod)
            else:
                self._empty_index[lhs] = prod

            for token in prod._rhs:
                if is_terminal(token):
                    self._lexical_index.setdefault(token, set()).add(prod)

    def start(self):
        """ Returns the start state of the grammar. """
        return self._start

    def productions(self, lhs=None, rhs=None):
        """ Returns a set of productions in the grammar, filtered by the lhs or the first entity in the rhs.

            :param lhs: Only return productions with the given lhs.
            :param rhs: Only return productions with the given first item in the rhs.
        """
        if not lhs and not rhs:
            return self._productions
        elif lhs and not rhs:
            return self._lhs_index.get(lhs, [])
        elif rhs and not lhs:
            return self._rhs_index.get(rhs, [])
        else:
            return [prod for prod in self._lhs_index.get(lhs,[]) \
                    if prod in self._rhs_index.get(rhs, [])]

    def check_coverage(self, tokens):
        """ Checks whether grammar cover the given list of tokens, and raises an exception if not.
        """
        missing = [tok for tok in tokens if not self._lexical_index.get(tok)]
        if missing:
            missing = ', '.join('{!r}'.format(w) for w in missing)
            raise ValueError("Grammar does not cover some of the "
                             "input words: {!r}".format(missing))

    def __repr__(self):
        return '<Grammar with {} productions>'.format(len(self._productions))

    def __str__(self):
        s = 'Grammar with {} productions'.format(len(self._productions))
        s += ' (start state = {!r})'.format(self._start)
        for prod in self._productions:
            s += '\n    {!s}'.format(prod)
        return s
    
#-----------------------------------------#

_ARROW_RE = re.compile(r'\s* -> \s*', re.VERBOSE)
_PROBABILITY_RE = re.compile(r'( \[ [\d\.]+ \] ) \s*', re.VERBOSE)
_TERMINAL_RE = re.compile(r'( "[^"]+" | \'[^\']+\' ) \s*', re.VERBOSE)
_DISJUNCTION_RE = re.compile(r'\| \s*', re.VERBOSE)
_NONTERM_RE = re.compile('( [\w/][\w/^<>-]* ) \s*', re.VERBOSE)


def parse_lwfg(rulesfile, lexfile):
    """
    Generates a Lexicalized Well-Founded Grammar after reading files containing
    the rules and the lexicon.

    The (input) files containing the rules and the lexicon have been split into two
    for a specific reason.  Namely, the difference in the content.  While the lexicon
    is nothing but a file containing the pre-terminal rules (e.g. ``noun -> `pencil'''),
    it also contains the semantic body and the head corresponding to the lexical item.

    On the other hand, the file containing the rules contains in addition the compositional
    constraints corresponding to each rule.

    These differences make it simpler to keep two separate files and parse them differently
    based on their structure.

    :param ``rulesfile'': the file containing the rules.
    :param ``lexfile'': the file containing the lexicon.
    """

    ##############################
    # Loading rules first
    ##############################
    rules_lines = [l.strip() for l in open(rulesfile).readlines()]
    lex_lines = [l.strip() for l in open(lexfile).readlines()]

    start = None
    productions = []

    for linenum, line in enumerate(rules_lines):
        # Comments start with #
        if not line or line.startswith('#'): continue

        # Directives start with %
        # The only implemented directive defines the start symbol.
        # If the directive is missing, the lhs of the first rule is taken to be the start symbol.
        if line[0] == '%':
            directive, args = line[1:].split(None,1)
            if directive == 'start':
                m = _NONTERM_RE.match(args, 0)
                start = LexNonterminal(m.group(1))
                if m.end() != len(args):
                    raise ValueError('Bad argument to start directive: {!s}'.format(line))
            else:
                raise ValueError('Bad directive: {!s}'.format(line))

        # Each rule is of the form X -> W+.
        # Thus, the first step in identifying is to detect the arrow ->.
        # Each rule is followed in the next line by a set of compositional constraints.
        # The constraints are represented in a way that makes it easy to generate a FeatStruct.
        # All one has to do is call nltk.featstruct.FeatStruct.
        if _ARROW_RE.findall(line):
            lhs, rhses = re.split(_ARROW_RE, line)
            if rhses == '':
                raise ValueError('Incomplete production rule: {!s}'.format(line))
            rhses = rhses.split()
            
            if len(rhses) > 1:
                for r in rhses:
                    if r[0] in "\'\"":
                        raise ValueError('Incorrectly formatted rule: {!s}'.format(line)) 

            lhs = LexNonterminal(lhs)
            for i, rhs in enumerate(rhses):
                if rhs[0] not in "\'\"":
                    rhses[i] = LexNonterminal(rhs)

            # now identifying the ontological constraints, and the syntagma
            try:
                phi_c = rules_lines[linenum+1]
                phi_c = FS(phi_c)

                lhs.setHead(phi_c['h'])
                for i, rhs in enumerate(rhses):
                    if is_nonterminal(rhs):
                        rhses[i].setHead(phi_c['h'+str(i+1)])
            except IndexError:
                raise ValueError('Rule {!s} is missing compositional constraints'.format(line))
            except KeyError:
                raise ValueError('Compositional constraints improperly formatted: {!s}'.format(rules_lines[linenum+1]))

            productions.append(LexProduction(lhs, rhses, phi_c))

    if not productions:
        raise ValueError('No productions found!')
    
    if not start:
        start = productions[0].lhs()

    ###############################
    # Now loading the lexicon
    ###############################

    # Like other rules, each rule in the lexicon contains an arrow ->.
    # The following line contains the compositional constraints, once again formatted
    # in a way that can directly be passed to the FeatStruct initializer.
    # The line after the compositional constraints contains the semantic body corresponding
    # to the lexical item.
    for linenum, line in enumerate(lex_lines):
        if _ARROW_RE.findall(line):
            lhs, rhses = re.split(_ARROW_RE, line)
            if rhses == '':
                raise ValueError('Incomplete production rule: {!s}'.format(line))
            rhses = rhses.split()
            
            if len(rhses) > 1:
               raise ValueError('Not a lexical rule: {!s}'.format(line)) 

            lhs = LexNonterminal(lhs)
            if not _TERMINAL_RE.match(rhses[0]):
                raise ValueError('Not a lexical rule: {!s}'.format(line))
            else:
                rhses[0] = rhses[0][1:-1]

            lhs.setString(rhses[0])

            # now identifying the ontological constraints, and the syntagma
            try:
                phi_c = lex_lines[linenum+1]
                phi_c = FS(phi_c)

                lhs.setHead(phi_c['h'])
            except IndexError:
                raise ValueError('Rule {!s} is missing compositional constraints'.format(line))
            except KeyError:
                raise ValueError('Compositional constraints improperly formatted: {!s}'.format(lex_lines[linenum+1]))

            # now the syntagma body
            try:
                body = lex_lines[linenum+2]
                
                lhs.setBody(OntoSeR(body))
            except IndexError:
                raise ValueError('Rule {!s} is missing its semantic body'.format(line))

            productions.append(LexProduction(lhs, rhses, phi_c))

    return LWFG(start, productions)


# -------------------------------- #

if __name__ == '__main__':
    gram = parse_lwfg('LWFG_TESTGRAM.txt', 'LWFG_TESTLEX.txt')

    ### random stuff, please ignore ###
    s = gram.start()
    sprods = gram.productions(s)
    prod = sprods[0]
    print prod
    n = prod.rhs()[0]
    print n, repr(n.head())

    nprods = gram.productions(n)
    for p in nprods:
        print p
        print repr(p.lhs().head())

        if n.head().unify(p.lhs().head()):
            print 'match'
        else:
            print 'no match'
