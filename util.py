import json

def getCredentials(file):

    credentials = json.load(open(file))
    return (credentials['username'], credentials['password'])

CONTRACTION_KEYS = '''ain't
aren't
can't
could've
couldn't
daren't
daresn't
dasn't
didn't
doesn't
don't
e'er
everyone's
finna
gimme
gonna
gotta
hadn't
hasn't
haven't
he'd
he'll
he's
how'd
how'll
how're
how's
I'd
I'll
I'm
I'm'a
I'm'o
I've
isn't
it'd
it'll
it's
let's
ma'am
mayn't
may've
mightn't
might've
mustn't
must've
needn't
ne'er
o'clock
o'er
ol'
oughtn't
shan't
she'd
she'll
she's
should've
shouldn't
somebody's
someone's
something's
that'll
that're
that's
that'd
there'd
there'll
there're
there's
these're
they'd
they'll
they're
they've
this's
those're
'tis
'twas
wasn't
we'd
we'd've
we'll
we're
we've
weren't
what'd
what'll
what're
what's
what've
when's
where'd
where're
where's
where've
which's
who'd
who'd've
who'll
who're
who's
who've
why'd
why're
why's
won't
would've
wouldn't
y'all
you'd
you'll
you're
you've
'''.lower().split('\n')

CONTRACTION_VALUES = '''are not
are not
cannot
could have
could not
dare not
dare not
dare not
did not
does not
do not 
ever
everyone is
fixing to
give me
going to
got to
had not
has not
have not
he would
 he will
he is
how did
how will
how are
how has 
I would
I will
I am
I am about to
I am going to
I have
is not
it would
it will
it is
let us
madam
may not
may have
might not
might have
must not
must have
need not
never
of the clock
over
old
ought not
shall not
she would
she will
she is
should have
should not
somebody is
someone is
something is
that will
that are
that is
that had
there would
there will
there are
there is
these are
they would
they will
they are
they have
this is
those are
it is
it was
was not
we had / we would
we would have
we will
we are
we have
were not
what did
what will
what are
what is
what have
when is
where did
where are
where is
where have
which is
who would 
who would have
who will
who are
who is
who have
why did
why are
why is
will not
would have
would not
you all
you would
you will
you are
you have'''.lower().split('\n')

CONTRACTION_MAP = list(zip(CONTRACTION_KEYS, CONTRACTION_VALUES))
