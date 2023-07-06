import operator
# stats = {'a': 1000, 'b': 3000, 'c': 100}
stats = {}
stats['a'] = [3000, 100, 100]
stats['b'] = [5000, 5000, 3000]
stats['c'] = [100, 3000, 5000]
print(max(stats.items(), key=operator.itemgetter(1))[0])
print(max(stats[k][0] for k in stats))