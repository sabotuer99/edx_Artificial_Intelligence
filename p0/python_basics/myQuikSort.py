def quiksort(arr):
  if(arr == None or len(arr) == 0):
    return []
  pivot = arr[0];
  left = [x for x in arr[1:] if x < pivot]
  right = [x for x in arr[1:] if x >= pivot]
  return quiksort(left) + [pivot] + quiksort(right)

# Main Function
if __name__ == '__main__':        
    print quiksort([5,3,4,7,8,9,2,5,7,4,5,2,3,3,3,5,9,1,2,6,5,4,7])