import pprint
import json

file_ = 'a6637a61fc42c734211f7df5089e88f859d0da5a.json'

f = open(file_, 'r')
text = json.loads(f.read())
pprint.pprint(text)
