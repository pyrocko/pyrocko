import sys, re

for fn in sys.argv[1:]:

    with open(fn, 'r') as f:
        s = f.read()

        xx = re.findall(r'([^\n]+)\s+\'\'\'(.*?)\'\'\'', s, re.M|re.S)
        for (obj, doc) in xx:
            s = re.findall('[^:`]\B(([`*])[a-zA-Z_][a-zA-Z0-9_]*\\2)\B', doc)
            if s:
                print '-'*50
                print fn, obj
                print '.'*50
                print doc
                print '.'*50
                print [ss[0] for ss in s]

#  for vim:
#  :s/\([^`:]\)\([`*]\)\([a-zA-Z0-9_]\+\)\2/\1``\3``/


