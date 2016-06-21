nums = [1,2,3,4,5,6]
oddNums = [x for x in nums if x % 2 == 1]
print oddNums
oddNumsPlusOne = [x+1 for x in nums if x % 2 ==1]
print oddNumsPlusOne

strs = ['duh', 'LONGSTRING', 'BroadSword', 'cat']
longstrs = [x.lower() for x in strs if len(x) > 5]
print longstrs
