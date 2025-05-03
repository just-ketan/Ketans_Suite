# Code_Snippets
Code Snippets of common coding questions
# every def is a new code 


Is the tree bst ?

    def is_bst(root):
        def helper(node, left, right):
            if not node:
                return True # we reached the end and there are no more violations
            if node.val <= left or node.val >= right:
                return False    # a value can never be lesser than left bound and greater than right bound in a substree
            
            return helper(node.left, left, node.val) and helper(node.right, node.val, right)
            #when we go down left, the upper bound will be the root node val    lc <= root
            # when we go down right, the lower bound will be the root node      rc >= root
        
        return is_bst(root, float('-inf'), float('inf'))

finid the first non repeating character

    def first_non_repeating_char(st:str)->str:
        mp = {}
        # or mp = defaultdict(int)
        
        for s in st:
            if s in mp:
                mp[s] += 1 
            else:
                mp[s] = 1 
        
        for s in st:
            if mp[s] == 1:
                return s.index()
        
        return -1

reverse a linked list

    def reverse_link_list():
        # to reverse a link list
        # 1(prev) -> 2(curr) -> 3(next) -> None
        # we will swap the links one at a time such that
        # 1 <- 2 -> 3
        # 1 <- 2 <- 3
        
        prev = None
        curr = head
        while curr:
            nxt = curr.next
            
            # main reversing logic
            curr.next = prev
            prev = curr
            curr = nxt
        return prev
        
        # 1(prev) -> 2(curr) -> 3(next) -> None
        # None (prev)
        # curr = head = 1 
        # nxt = curr.next   =>      nxt = 2
        # curr.next = prev  =>      1 -> None
        # prev = curr       => prev = 1 
        # curr = nxt        => curr = 2
            
        # nxt = curr.next   =>      nxt = 3
        # curr.next = prev  =>      2 -> 1 -> None
        # prev = curr       => prev = 2 
        # curr = nxt        => curr = 3
        
        # nxt = curr.next   =>      nxt = None
        # curr.next = prev  =>      3 -> 2 -> 1 -> None
        # prev = curr       => prev = 3 
        # curr = nxt        => curr = None      --> while loop breaks
        
        # return prev -> returns the new head, that is 3
        # 3 -> 2 -> 1 is returned

implement quicksort

    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        
        pivot = len(arr) // 2
        left = [ l for l in arr if l < pivot]       # list of entries less than pivot
        middle = [ m for m in arr if m == pivot ]   # list of entries equal to pivot
        right = [ r for r in arr if r > pivot]      # list of entries greater than pivot
        
        return quicksort(left) + middle + quicksort(right)      # recursive call to invoke the call stack

two sum

    def twosum(arr,target):
        mp = {} 
        for i in range(len(arr)):
            mp[arr[i]] = i      # mapping number to its index
        
        for i in range(len(arr)):
            comp = target - arr[i]      # finding complement, 2+3 = 5, them there must be a 5-2 in mp if we finding twosum for 2 
            if comp in mp:
                return [i, mp[comp]]
        
        return [-1,-1]      # not found

max subarray sum

    def maxsubarraysum(arr,l,r):
        # need to find sum of subarrays starting from left, inclusive of r
        # we implement prefix sum
        pfx = []
        summ = 0
        for i in arr:
            summ += i
            pfx.append(summ)
            # at every index i, we get the summ till ith element
        
        return (pfx[r] - pfx[l])        # this gives the exact sum in O(n) time
    
    
    
<----------Divide and Conquer mergesort --------------->
merge sort

    def mergesort(arr):
        if len(arr) <= 1: return arr
        
        mid = len(arr) // 2 
        
        left = mergesort(arr[:mid])
        right = mergesort(arr[mid:])        # recursive calls to break down left and right segments till unit block of element remains
        
        return merger(left,right)       # will have 2^n merge calls in recursive call stack, on the way up, it will build back the sorted array 
            
    def merge(left,right):
        res = []
        i = j= 0
        while i < len(left) and right < len(right):
            if left[i] <= right[j]:
                res.append(left[i])
                i+=1 
            else:
                res.append(right[j])
                j += 1 
        
        res.extend(left[i:])        # for leftover elements in left and right, it extends res and copies values into res 
        res.extend(right[j:])
        
        return res
            
            
selection sort

    def selection_sort(arr):        # repeatedly swap smaller elements with larger elements 
        for i in range(len(arr)):
            min_idx = i
            for j in range(i+1, len(arr)):
                if arr[j] < arr[min_idx]:       # if element in subarray has smaller value then swap
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr
            
bubble sort

    def bubblesort(arr):
        for i in range(len(arr)):
            swapped = False
            for j in range(len(arr) - i - 1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
                    swapped = True
            if not swapped:
                break
        return arr
        
insertion sort

    def insertionsort(arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1 
            
            while j >= 0 and key < arr[j]:
                arr[j+1] = arr[j] 
                j-=1 
            arr[j+1] = key
        return arr
    
        '''Initial Array: [5, 2, 9, 1, 5]
        
        Start with the 2nd element (i = 1)
        key = 2 (current element to insert).
        Compare 2 with 5 (the sorted subarray [5] to its left).
        Since 2 < 5, shift 5 right → [5, 5, 9, 1, 5].
        Insert key at the correct position → [2, 5, 9, 1, 5].
        
        Next element (i = 2)
        key = 9.
        Compare 9 with 5 (sorted subarray [2, 5]).
        9 > 5 → no shifts needed → [2, 5, 9, 1, 5].
        
        Next element (i = 3)
        key = 1.
        Compare 1 with 9 → shift 9 right → [2, 5, 9, 9, 5].
        Compare 1 with 5 → shift 5 right → [2, 5, 5, 9, 5].
        Compare 1 with 2 → shift 2 right → [2, 2, 5, 9, 5].
        Insert key → [1, 2, 5, 9, 5]
        
        Final element (i = 4)
        key = 5.
        Compare 5 with 9 → shift 9 right → [1, 2, 5, 9, 9].
        Compare 5 with 5 (no shift, since 5 == 5).
        Insert key → [1, 2, 5, 5, 9].'''
    
<---------- Dynamic Programming [memoization(top - down) / Tabulation(bottom - up)] --------------->

return nth fibonacci number

        def fibonacci(self, n: int) -> int:
            # bottom up dp 
            # base cases
            if n == 0:
                return 0
            if n == 1 or n == 2:
                return 1

        dp = [0]*(n+1)
        dp[0] = 0
        dp[1] = 1
        dp[2] = 1
        for i in range(3,len(dp)):
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[n]
    
    
climbing stairs :  can climb one or two at a time, how many steps to reach top

    def climbing(self, n:int) -> int:
        
        if n <= 3:
            # if stairs is < 3, there are n ways
            # 1 -> 1 -> 1 == 3
            # 1 -> 2 == 3
            # 2 -> 1 == 3

        dp = [0] * (n+1)
        dp[0] = 0   # represents base case of being at position 0
        dp[1] = 1   # can climb one stair to reach dp[1]
        dp[2] = 2   # can climb 2 stairs to rach dp[2]
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[i]
        # this takes O(n) space for dp buildup
        
        # space optimised for O(1)
        
        if n <= 3:
            return n
        
        a = 1 
        b = 2
        for _ in range(3, n+1):
            a, b = b, a+b 
        return b
        
        # we basically have 2 pointers, a and b that move forward and b is always 1 jump ahead.
        # no of jumps b can do would be all the  jumps it could do earlier ( given by position a) + number of jumps he can do with 2 choice
        
0/1 knapsack
    # we are given n items with weight[i] and value[i]
    # and maxweight as w 
    # choose items such that sum(weights[i]) < maxweight and sum(value[i]) is maximised
    
    def zerooneknapsack(w, wt, val, n):
        # we need a 2D dp table where each index i has an index j that represents max of sum(values) for that index i 
        dp[][] = [[0 for _ in range(w+1)] for _ in range(n+1)]
        # we always create n+1 size dp table to account for base case of 0
        
        # base cases
        # dp[0][j] -> represents no items --> 0
        # dp[i][0] --> represents no capacity --> 0
        
        for i in range(1, n+1):         # from 1 to n inclusive of n 
            for j in range(1, w+1):     # from 1 to w inclusive of w
                if wt[i-j] > j:
                    # cant take this item
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j]  = max(dp[i-1][j], (val[i-1] + dp[i-1][j - wt[i-1]] ))
                    
        return dp[n][w]
        
space optimised 0/1 knapsack 
    
    def knapsack(w,wt,val,n):
        dp = [0]*(w+1)
        # dp[i] reperesents the maximum capacity the knapsack of size i can have
        # initally 0 as with 0 size, we can store 0 items
        
        for i in range(n):
            # for each item i we decide whether we take it or not
            # we basically chose the max of the two options
            
            for j in range(w, wt[i]-1, -1):
                # we traverse the remaining in reverse to prevent reuse of same elements multiple times
                # iterate from maxxweight till the maximum possible weight at index j
                dp[j] = max(dp[j], (val[i] + dp[j - wt[i]]))
                
                # if this chose dp[j] -> we did not select current j
                # if this chose to include the current entry
                    # we need to add the value given at this index i
                    # subtract the weight(wt[i]) from current capacity j (dp[j])
                    # remaining capacity is dp[j - wt[i]] --> we contribute this to current capacity
        return dp[w]
                    
                    
        
        
    
Longest Common Subsequence
    given a string str1 and str2, return the len of longest common Subsequence
    
    def lcs(str1, str2);
        m,n = len(str1), len(str1)
        
        dp[][] = [[0]*(n+1) for _ in range(m+1)]     # n columns, m rows
        # rows represent str1 len ( 0 to m)
        # cols represent str2 len (0 to n)
        
        # base case dp[0][1] = dp[1][0] = 0     comparing with an empty string
        for i in range(1, m+1):
            for j in range(1, n+1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                    # the characters match, thus it contributes to the maxlength
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    # we either skip a character from str1 or from str2
                    # we get the max length possible from each action
        return dp[m][n]
                    
                    
                    
                    
coin exchange
    given an amount, and a set of coins, return possible combinations of coins that make up amount
    
    def coinex(coins, amount):
        dp = [float('inf')]*(amount+1)
        # dp[i] represents the min coins required to make a sum of i from given coins
        dp[0] = 0   # 0 amount can be reallised by taking 0 coins from coins list
        
        for coin in coins:
            for i in range(coin, amount+1):     # from current coin to max value possible, inclusive of amount
                dp[i] = min(dp[i], dp[i - coin] + 1)
        
        return dp[amount] if dp[amount] != float('inf') else -1 
                    
                    
                
<-------------------------Trees -------------------------------->

# check if binary tree is BST 
    
    def isbst(node:TreeNode, minn = float('-inf'), maxx = float('inf')) -> bool:
        # we basically need to make sure, left is < root and right is > root
        # the deafult minn is -inf and maxx is inf
        
        if not node:
            return True     # we have reached the end of the TreeNode
        
        if not (minn < node.val < maxx):
            return False
        
        return isbst(node.left, minn, node.val) and isbst(node.right, node.val, maxx)
        # for left sub tree, the max allowed value would be the root
        # for right sub tree, the min allowed value would be the root
        # we recusively traverse the tree, if this setup fails, the tree is not BST
    

# Height/Depth of BST
    
    def height(node:TreeNode) -> int:
        if not node:
            return -1 # no children
        
        return 1 + max(height(node.left), height(node.right))



# Lowest common ancester in BST
    
    def lca(root:TreeNode, node1,node2) -> TreeNode:
        # bst property is that lst <= root <= rst
        while root:
            if node1.val < root.val and node2.val < root.val: #both are smaller than the current node
                root = root.left
            elif node1.val > root.val and node2.val > root.val: # both lie in right substree
                root = root.right
            else:       # they both lie somewhere in this subtree, and this this is LCA
                return root

    # for a generic tree / not necessarily a bst
    
    def lcs(node, n1, n2):
        if not node or node == n1 or node == n2:
            return node     #hit the base case 
            
        # must traverse the entire tree to find the matches
        left = lcs(node.left,n1,n2)
        right = lcs(node.right,n1,n2)
        
        if left and right:
            return node     # both n1 and n2 lie in different subtrees
        
        return left or right    # return non-nul subtree
        

# level order traversal
    def lot(root):
        q = deque([root])
        res = []
        
        while q:
            qlen(len(q))
            curr = []
            for i in range(qlen):
                node = q.popleft()
                if node:
                    curr.append(node)
                    q.append(node.left)
                    q.append(node.right)
            if curr:
                res.append(curr)
        
        return res

# Serialise and deserealise a tree
    # objective is to convert a tree to string
    # build a tree using this string 

    def serialise(root):
        if not root:
            return "null"
        res = []    # list to represent Level order traversal
        q = deque([root])
        while q:
            node = q.popleft()
            if node:
                res.append(str(node.val))
                q.append(node.left)
                q.append(node.right)
            else:
                res.append("null")
        return ",".join(res)
        
    
    def deserealise(data):
        if data== "null":
            return None
        
        values = data.split(",")
        root = TreeNode(int(values[0]))
        q = deque([root])
        i=1 
        
        while q and i < len(values):
            node = q.popleft()
            if values[i] != "null":
                node.left = TreeNode(int(values[i]))
                q.append(node.left)
            i+=1 
            if values[i] != "null":
                node.right = TreeNode(int(values[i]))
                q.append(node.right)
            i+=1
        return root
