---
layout: post
title: Daily leetcode challenge
---

# Daily leetcode challenge
You can join me and discuss in the Telegram channel [https://t.me/leetcode_daily_unstoppable](https://t.me/leetcode_daily_unstoppable)

# 21.01.2022
[93. Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/description/) medium

[https://t.me/leetcode_daily_unstoppable/92](https://t.me/leetcode_daily_unstoppable/92)

[blog post](https://leetcode.com/problems/restore-ip-addresses/solutions/3080471/kotlin-dfs-backtracking/)

```kotlin 
    fun restoreIpAddresses(s: String): List<String> {
        val res = mutableSetOf<String>()
        fun dfs(pos: Int, nums: MutableList<Int>) {
            if (pos == s.length) {
                if (nums.size == 4) res += nums.joinToString(".")
                return
            }
            if (s[pos] == '0') {
                nums += 0
                dfs(pos + 1, nums)
                nums.removeAt(nums.lastIndex)
            } else {
                var i = pos
                for (i in pos..s.lastIndex) {
                    val num = s.substring(pos, i + 1).toInt()
                    if (num > 255) break
                    nums += num
                    dfs(i + 1, nums)
                    nums.removeAt(nums.lastIndex)
                }
            }
        }
        dfs(0, mutableListOf())
        return res.toList()
    }
```
So, the size of the problem is small. We can do full DFS.
At every step, choose either take a number or split. Add to the solution if the result is good.

* use set for results
* use backtracking to save some space

Space: O(2^N), Time: O(2^N)

# 20.01.2022
[491. Non-decreasing Subsequences](https://leetcode.com/problems/non-decreasing-subsequences/description/) medium

[https://t.me/leetcode_daily_unstoppable/91](https://t.me/leetcode_daily_unstoppable/91)

[blog post](https://leetcode.com/problems/non-decreasing-subsequences/solutions/3075577/kotlin-backtraking-set/)

```kotlin 
    fun findSubsequences(nums: IntArray): List<List<Int>> {
        val res = mutableSetOf<List<Int>>()
        fun dfs(pos: Int, currList: MutableList<Int>) {
            if (currList.size > 1) res += currList.toList()
            if (pos == nums.size) return
            val currNum = nums[pos]
            //not add
            dfs(pos + 1, currList)
            //to add
            if (currList.isEmpty() || currList.last()!! <= currNum) {
                currList += currNum
                dfs(pos + 1, currList)
                currList.removeAt(currList.lastIndex)
            }
        }
        dfs(0, mutableListOf())
        return res.toList()
    }
```

Notice the size of the problem, we can do a brute force search for all solutions. Also, we only need to store the unique results, so we can store them in a set.

* we can reuse pre-filled list and do backtracking on the return from the DFS.
 
Space: O(2^N) to store the result, Time: O(2^N) for each value we have two choices, and we can build a binary tree of choices with the 2^n number of elements. 
 
# 19.01.2022
[974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/) medium

[https://t.me/leetcode_daily_unstoppable/90](https://t.me/leetcode_daily_unstoppable/90)

[blog post](https://leetcode.com/problems/subarray-sums-divisible-by-k/solutions/3073473/kotlin-prefix-sum-and-remainders/)

```kotlin 
    fun subarraysDivByK(nums: IntArray, k: Int): Int {
        // 4 5 0 -2 -3 1    k=5   count
        // 4                4:1   0
        //   9              4:2   +1
        //     9            4:3   +2
        //       7          2:1   
        //          4       4:4   +3
        //             5    0:2   +1
        // 2 -2 2 -4       k=6
        // 2               2:1
        //    0            0:2    +1
        //      2          2:2    +1
        //        -2       2:3    +2
        // 1 2 13 -2 3  k=7
        // 1
        //   3
        //     16
        //        14
        //          17 (17-1*7= 10, 17-2*7=3, 17-3*7=-4, 17-4*7 = -11)
        val freq = mutableMapOf<Int, Int>()
        freq[0] = 1
        var sum = 0 
        var res = 0
        nums.forEach {
            sum += it
            var ind = (sum % k)
            if (ind < 0) ind += k
            val currFreq = freq[ind] ?: 0
            res += currFreq
            freq[ind] = 1 + currFreq
        }
        return res
    }
```
We need to calculate a running sum. 
For every current sum, we need to find any subsumes that are divisible by k, so `sum[i]: (sum[i] - sum[any prev]) % k == 0`. 
Or, `sum[i] % k == sum[any prev] % k`. 
Now, we need to store all `sum[i] % k` values, count them and add to result.

We can save frequency in a map, or in an array [0..k], because all the values are from that range.

Space: O(N), Time: O(N)

# 18.01.2022
[918. Maximum Sum Circular Subarray](https://leetcode.com/problems/maximum-sum-circular-subarray/description/) medium

[https://t.me/leetcode_daily_unstoppable/89](https://t.me/leetcode_daily_unstoppable/89)

[blog post](https://leetcode.com/problems/maximum-sum-circular-subarray/solutions/3069120/kotlin-invert-the-problem/)

```kotlin 
    fun maxSubarraySumCircular(nums: IntArray): Int {
        var maxEndingHere = 0
        var maxEndingHereNegative = 0
        var maxSoFar = Int.MIN_VALUE
        var total = nums.sum()
        nums.forEach {
            maxEndingHere += it
            maxEndingHereNegative += -it
            maxSoFar = maxOf(maxSoFar, maxEndingHere, if (total == -maxEndingHereNegative) Int.MIN_VALUE else total+maxEndingHereNegative)
            if (maxEndingHere < 0) {
                maxEndingHere = 0
            }
            if (maxEndingHereNegative < 0) {
                maxEndingHereNegative = 0
            }
        }
        return maxSoFar
    }
```

Simple Kadane's Algorithm didn't work when we need to keep a window of particular size. 
One idea is to invert the problem and find the minimum sum and subtract it from the total.

One corner case:
* we can't subtract all the elements when checking the negative sum.

Space: O(1), Time: O(N)

# 17.01.2022
[926. Flip String to Monotone Increasing](https://leetcode.com/problems/flip-string-to-monotone-increasing/description/) medium

[https://t.me/leetcode_daily_unstoppable/88](https://t.me/leetcode_daily_unstoppable/88)

[blog post](https://leetcode.com/problems/flip-string-to-monotone-increasing/solutions/3062530/kotlin-dp/)

```kotlin 
    fun minFlipsMonoIncr(s: String): Int {
        // 010110  dp0  dp1    min
        // 0       0    0      0
        //  1      1    0      1
        //   0     1    1      1
        //    1    2    1      1
        //     1   3    1      1
        //      0  3    2      2
        var dp0 = 0
        var dp1 = 0

        for (i in 0..s.lastIndex) {
            dp0 = if (s[i] == '0') dp0 else 1 + dp0
            dp1 = if (s[i] == '1') dp1 else 1 + dp1
            if (dp0 <= dp1) dp1 = dp0
        }
        
        return minOf(dp0, dp1)
    }
```

We can propose the following rule: let's define `dp0[i]` is a min count of flips from `1` to `0` in the `0..i` interval. 
Let's also define `dp1[i]` is a min count of flips from `0` to `1` in the `0..i` interval. 
We observe that `dp0[i] = dp0[i-1] + (flip one to zero? 1 : 0)` and `dp1[i] = dp1[i-1] + (flip zero to one? 1 : 0)`. 
One special case: if on the interval `0..i` one-to-zero flips count is less than zero-to-one then we prefer to flip everything to zeros, and `dp1[i]` in that case becomes `dp0[i]`.

Just write down what is described above.
* dp arrays can be simplified to single variables.

Space: O(1), Time: O(N)

# 16.01.2022
[57. Insert Interval](https://leetcode.com/problems/insert-interval/description/) medium

[https://t.me/leetcode_daily_unstoppable/87](https://t.me/leetcode_daily_unstoppable/87)

[blog post](https://leetcode.com/problems/insert-interval/solutions/3057540/kotlin-one-pass/)

```kotlin 
    fun insert(intervals: Array<IntArray>, newInterval: IntArray): Array<IntArray> {
        val res = mutableListOf<IntArray>()
        var added = false
        fun add() {
            if (!added) {
                added = true
                if (res.isNotEmpty() && res.last()[1] >= newInterval[0]) {
                    res.last()[1] = maxOf(res.last()[1], newInterval[1])
                } else res += newInterval
            }
        }
        intervals.forEach { interval -> 
            if (newInterval[0] <= interval[0]) add()
            
            if (res.isNotEmpty() && res.last()[1] >= interval[0]) {
                res.last()[1] = maxOf(res.last()[1], interval[1])
            } else  res += interval
        }
        add()
       
        return res.toTypedArray()
    }
```

There is no magic, just be careful with corner cases.

Make another list, and iterate interval, merging them and adding at the same time.
* don't forget to add `newInterval` if it is not added after iteration.

Space: O(N), Time: O(N)

# 15.01.2022
[2421. Number of Good Paths](https://leetcode.com/problems/number-of-good-paths/) hard

[https://t.me/leetcode_daily_unstoppable/86](https://t.me/leetcode_daily_unstoppable/86)

[blog post](https://leetcode.com/problems/number-of-good-paths/solutions/3054534/kotlin-union-find-was-hard/)

```kotlin 
    fun numberOfGoodPaths(vals: IntArray, edges: Array<IntArray>): Int {
        if (edges.size == 0) return vals.size
        edges.sortWith(compareBy(  { maxOf( vals[it[0]], vals[it[1]] ) }  ))
        val uf = IntArray(vals.size) { it }
        val freq = Array(vals.size) { mutableMapOf(vals[it] to 1) }
        fun find(x: Int): Int {
            var p = x
            while (uf[p] != p) p = uf[p]
            uf[x] = p
            return p
        }
        fun union(a: Int, b: Int): Int {
            val rootA = find(a)
            val rootB = find(b)
            if (rootA == rootB) return 0
            uf[rootA] = rootB
            val vMax = maxOf(vals[a], vals[b]) // if we connect tree [1-3] to tree [2-1], only `3` matters
            val countA = freq[rootA][vMax] ?:0
            val countB = freq[rootB][vMax] ?:0
            freq[rootB][vMax] = countA + countB
            return countA * countB
        }
        return edges.map { union(it[0], it[1])}.sum()!! + vals.size
    }
```

The naive solution with single DFS and merging frequency maps gives TLE. 
Now, use hint, and they tell you to sort edges and use Union-Find :) 
The idea is to connect subtrees, but walk them from smallest to the largest of value. 
When we connect two subtrees, we look at the maximum of each subtree. 
The minimum values don't matter because the path will break at the maximums by definition of the problem.

Use IntArray for Union-Find, and also keep frequencies maps for each root.

Space: O(NlogN), Time: O(N)

# 14.01.2022
[1061. Lexicographically Smallest Equivalent String](https://leetcode.com/problems/lexicographically-smallest-equivalent-string/description/) medium

[https://t.me/leetcode_daily_unstoppable/85](https://t.me/leetcode_daily_unstoppable/85)

[blog post](https://leetcode.com/problems/lexicographically-smallest-equivalent-string/solutions/3049304/kotlin-uniton-find/)

```kotlin 
    fun smallestEquivalentString(s1: String, s2: String, baseStr: String): String {
        val uf = IntArray(27) { it }
        fun find(ca: Char): Int {
            val a = ca.toInt() - 'a'.toInt()
            var x = a
            while (uf[x] != x) x = uf[x]
            uf[a] = x
            return x
        }
        fun union(a: Char, b: Char) {
            val rootA = find(a)
            val rootB = find(b)
            if (rootA != rootB) {
                val max = maxOf(rootA, rootB)
                val min = minOf(rootA, rootB)
                uf[max] = min
            }
        }
        for (i in 0..s1.lastIndex) union(s1[i], s2[i])
        return baseStr.map { (find(it) + 'a'.toInt()).toChar() }.joinToString("")
    }
```

We need to find connected groups, the best way is to use the Union-Find.

Iterate over strings and connect each of their chars.
* to find a minimum, we can select the minimum of the current root.

Space: O(N) for storing a result, Time: O(N)

# 13.01.2022
[2246. Longest Path With Different Adjacent Characters](https://leetcode.com/problems/longest-path-with-different-adjacent-characters/description/) hard

[https://t.me/leetcode_daily_unstoppable/84](https://t.me/leetcode_daily_unstoppable/84)

[blog post](https://leetcode.com/problems/longest-path-with-different-adjacent-characters/solutions/3046179/kotlin-build-graph-dfs/)

```kotlin 
    fun longestPath(parent: IntArray, s: String): Int {
        val graph = mutableMapOf<Int, MutableList<Int>>()
        for (i in 1..parent.lastIndex) 
            if (s[i] != s[parent[i]]) graph.getOrPut(parent[i], { mutableListOf() }) += i
        
        var maxLen = 0
        fun dfs(curr: Int): Int {
            parent[curr] = curr
            var max1 = 0
            var max2 = 0
            graph[curr]?.forEach { 
                val childLen = dfs(it) 
                if (childLen > max1) {
                    max2 = max1
                    max1 = childLen
                } else if (childLen > max2) max2 = childLen
            }
            val childChainLen = 1 + (max1 + max2)
            val childMax = 1 + max1
            maxLen = maxOf(maxLen, childMax, childChainLen)
            return childMax
        }
        for (i in 0..parent.lastIndex) if (parent[i] != i) dfs(i)

        return maxLen
    }
```

Longest path is a maximum sum of the two longest paths of the current node.

Let's build a graph and then recursively iterate it by DFS. We need to find two largest results from the children DFS calls.
* make `parent[i] == i` to store a `visited` state

Space: O(N), Time: O(N), in DFS we visit each node only once.

# 12.01.2022
[1519. Number of Nodes in the Sub-Tree With the Same Label](https://leetcode.com/problems/number-of-nodes-in-the-sub-tree-with-the-same-label/description/) medium

[https://t.me/leetcode_daily_unstoppable/83](https://t.me/leetcode_daily_unstoppable/83)

[blog post](https://leetcode.com/problems/number-of-nodes-in-the-sub-tree-with-the-same-label/solutions/3039078/kotlin-build-graph-count-by-dfs/)

```kotlin 
fun countSubTrees(n: Int, edges: Array<IntArray>, labels: String): IntArray {
	val graph = mutableMapOf<Int, MutableList<Int>>()
	edges.forEach { (from, to) ->
		graph.getOrPut(from, { mutableListOf() }) += to
		graph.getOrPut(to, { mutableListOf() }) += from
	}
	val answer = IntArray(n) { 0 }
	fun dfs(node: Int, parent: Int, counts: IntArray) {
		val index = labels[node].toInt() - 'a'.toInt()
		val countParents = counts[index]
		counts[index]++
		graph[node]?.forEach {
			if (it != parent) {
				dfs(it, node, counts)
			}
		}
		answer[node] = counts[index] - countParents
	}
	dfs(0, 0, IntArray(27) { 0 })
	return answer
}
```
First, we need to build a graph. Next, just do DFS and count all `'a'..'z'` frequencies in the current subtree.

For building a graph let's use a map, and for DFS let's use a recursion.
* use `parent` node instead of the visited set
* use in-place counting and subtract `count before`

Space: O(N), Time: O(N)

# 11.01.2022
[1443. Minimum Time to Collect All Apples in a Tree](https://leetcode.com/problems/minimum-time-to-collect-all-apples-in-a-tree/description/) medium

[https://t.me/leetcode_daily_unstoppable/82](https://t.me/leetcode_daily_unstoppable/82)

[blog post](https://leetcode.com/problems/minimum-time-to-collect-all-apples-in-a-tree/solutions/3036411/kotlin-build-tree-and-count-paths-to-parents/)

```kotlin 
    fun minTime(n: Int, edges: Array<IntArray>, hasApple: List<Boolean>): Int {
        val graph = mutableMapOf<Int, MutableList<Int>>()
        edges.forEach { (from, to) ->
            graph.getOrPut(to, { mutableListOf() }) += from
            graph.getOrPut(from, { mutableListOf() }) += to
        }

        val queue = ArrayDeque<Int>()
        queue.add(0)
        val parents = IntArray(n+1) { it }
        while (queue.isNotEmpty()) {
            val node = queue.poll()
            graph[node]?.forEach {
                if (parents[it] == it && it != 0) {
                    parents[it] = node
                    queue.add(it)
                }
            }
        }
        var time = 0
        hasApple.forEachIndexed { i, has ->
            if (has) {
                var node = i
                while (node != parents[node]) {
                    val parent = parents[node]
                    parents[node] = node
                    node = parent
                    time++
                }
            }
        }
        return time * 2
    }
```

We need to count all paths from apples to 0-node and don't count already walked path.
* notice, that problem definition doesn't state the order of the edges in `edges` array. We need to build the tree first.

First, build the tree, let it be a `parents` array, where `parent[i]` is a parent of the `i`. 
Walk graph with DFS and write the parents. 
Next, walk `hasApple` list and for each apple count parents until reach node `0` or already visited node. 
To mark a node as visited, make it the parent of itself.

Space: O(N), Time: O(N)

# 10.01.2022
[100. Same Tree](https://leetcode.com/problems/same-tree/description/) easy

[https://t.me/leetcode_daily_unstoppable/81](https://t.me/leetcode_daily_unstoppable/81)

[blog post](https://leetcode.com/problems/same-tree/solutions/3028835/kotlin-recursive/)

```kotlin 
fun isSameTree(p: TreeNode?, q: TreeNode?): Boolean =  p == null && q == null || 
            p?.`val` == q?.`val` && isSameTree(p?.left, q?.left) && isSameTree(p?.right, q?.right)
```
Check for the current node and repeat for the children.
Let's write one-liner

Space: O(logN) for stack, Time: O(n)

# 9.01.2022
[144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/description/) easy

[https://t.me/leetcode_daily_unstoppable/80](https://t.me/leetcode_daily_unstoppable/80)

[blog post](https://leetcode.com/problems/binary-tree-preorder-traversal/solutions/3023310/kotlin-morris-stack-recursive/)

```kotlin 
class Solution {
    fun preorderTraversal(root: TreeNode?): List<Int> {
        val res = mutableListOf<Int>()
        var node = root
        while(node != null) {
            res.add(node.`val`)
            if (node.left != null) {
                if (node.right != null) {
                    var rightmost = node.left!!
                    while (rightmost.right != null) rightmost = rightmost.right
                    rightmost.right = node.right
                }
                node = node.left
            } else if (node.right != null) node = node.right
            else node = null
        }
        return res
    }
    fun preorderTraversalStack(root: TreeNode?): List<Int> {
        val res = mutableListOf<Int>()
        var node = root
        val rightStack = ArrayDeque<TreeNode>()
        while(node != null) {
            res.add(node.`val`)
            if (node.left != null) {
                if (node.right != null) {
                    rightStack.addLast(node.right!!) // <-- this step can be replaced with Morris
                    // traversal.
                }
                node = node.left
            } else if (node.right != null) node = node.right
            else if (rightStack.isNotEmpty()) node = rightStack.removeLast()
            else node = null
        }
        return res
    }
    fun preorderTraversalRec(root: TreeNode?): List<Int> = mutableListOf<Int>().apply {
        root?.let {
            add(it.`val`)
            addAll(preorderTraversal(it.left))
            addAll(preorderTraversal(it.right))
        }
    }
        
}
```
Recursive solution is a trivial. For stack solution, we need to remember each `right` node. Morris' solution use the tree modification to save each `right` node in the rightmost end of the left subtree.
Let's implement them all.

Space: O(logN) for stack, O(1) for Morris', Time: O(n)

# 8.01.2022
[149. Max Points on a Line](https://leetcode.com/problems/max-points-on-a-line/) hard

[https://t.me/leetcode_daily_unstoppable/79](https://t.me/leetcode_daily_unstoppable/79)

[blog post](https://leetcode.com/problems/max-points-on-a-line/solutions/3018971/kotlin-linear-algebra-n-2/)

```kotlin 
    fun maxPoints(points: Array<IntArray>): Int {
        if (points.size == 1) return 1
        val pointsByTan = mutableMapOf<Pair<Double, Double>, HashSet<Int>>()
        fun gcd(a: Int, b: Int): Int {
            return if (b == 0) a else gcd(b, a%b)
        }
        for (p1Ind in points.indices) {
            val p1 = points[p1Ind]
            for (p2Ind in (p1Ind+1)..points.lastIndex) {
                val p2 = points[p2Ind]
                val x1 = p1[0]
                val x2 = p2[0]
                val y1 = p1[1]
                val y2 = p2[1]
                var dy = y2 - y1
                var dx = x2 - x1
                val greatestCommonDivider = gcd(dx, dy)
                dy /= greatestCommonDivider
                dx /= greatestCommonDivider
                val tan = dy/dx.toDouble()
                val b = if (dx == 0) x1.toDouble() else (x2*y1 - x1*y2 )/(x2-x1).toDouble()
                val line = pointsByTan.getOrPut(tan to b, { HashSet() })
                line.add(p1Ind)
                line.add(p2Ind)
            }
        }
        return pointsByTan.values.maxBy { it.size }?.size?:0
    }
```
Just do the linear algebra to find all the lines through each pair of points.
Store `slope` and `b` coeff in the hashmap. Also, compute `gcd` to find precise slope. In this case it works for `double` precision slope, but for bigger numbers we need to store `dy` and `dx` separately in `Int` precision.

Space: O(n^2), Time: O(n^2)

# 7.01.2022
[134. Gas Station](https://leetcode.com/problems/gas-station/description/) medium

[https://t.me/leetcode_daily_unstoppable/78](https://t.me/leetcode_daily_unstoppable/78)

[blog post](https://leetcode.com/problems/gas-station/solutions/3013707/kotlin-greedy/)

```kotlin 
    fun canCompleteCircuit(gas: IntArray, cost: IntArray): Int {
        var sum = 0
        var minSum = gas[0]
        var ind = -1
        for (i in 0..gas.lastIndex) {
            sum += gas[i] - cost[i]
            if (sum < minSum) {
                minSum = sum
                ind = (i+1) % gas.size
            }
        }
        return if (sum < 0) -1 else ind
    }
```

We can start after the station with the minimum `decrease` in gasoline.
![image.png](https://assets.leetcode.com/users/images/252d5b9e-b28b-4306-95bc-b37c1afed1b9_1673095767.9064982.png)
Calculate running gasoline volume and find the minimum of it. If the total net gasoline is negative, there is no answer.

Space: O(1), Time: O(N)

# 6.01.2022
[1833. Maximum Ice Cream Bars](https://leetcode.com/problems/maximum-ice-cream-bars/description/) medium

[https://t.me/leetcode_daily_unstoppable/77](https://t.me/leetcode_daily_unstoppable/77)

[blog post](https://leetcode.com/problems/maximum-ice-cream-bars/solutions/3007210/kotlin-greedy/)

```kotlin 
    fun maxIceCream(costs: IntArray, coins: Int): Int {
       costs.sort() 
       var coinsRemain = coins
       var iceCreamCount = 0
       for (i in 0..costs.lastIndex) {
           coinsRemain -= costs[i]
           if (coinsRemain < 0) break
           iceCreamCount++
       }
       return iceCreamCount
    }
```
The `maximum ice creams` would be if we take as many `minimum costs` as possible
Sort the `costs` array, then greedily iterate it and buy ice creams until all the coins are spent.

Space: O(1), Time: O(NlogN) (there is also O(N) solution based on count sort)

# 5.01.2022
[452. Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/description/) medium

[https://t.me/leetcode_daily_unstoppable/75](https://t.me/leetcode_daily_unstoppable/75)

[blog post](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/solutions/3002258/kotlin-sort-line-sweep/)

```kotlin 
    fun findMinArrowShots(points: Array<IntArray>): Int {
        if (points.isEmpty()) return 0
        if (points.size == 1) return 1
        Arrays.sort(points, Comparator<IntArray> { a, b -> 
            if (a[0] == b[0]) a[1].compareTo(b[1]) else a[0].compareTo(b[0]) })
        var arrows = 1
        var arrX = points[0][0]
        var minEnd = points[0][1]
        for (i in 1..points.lastIndex) {
            val (start, end) = points[i]
            if (minEnd < start) {
                arrows++
                minEnd = end
            }
            if (end < minEnd) minEnd = end
            arrX = start
        }
        return arrows
    }
```
The optimal strategy to achieve the minimum number of arrows is to find the maximum overlapping intervals. For this task, we can sort the points by their `start` and `end` coordinates and use line sweep technique. Overlapping intervals are separate if their `minEnd` is less than `start` of the next interval. `minEnd` - the minimum of the `end`'s of the overlapping intervals.
Let's move the arrow to each `start` interval and fire a new arrow if this `start` is greater than `minEnd`.
* for sorting without Int overflowing, use `compareTo` instead of subtraction
* initial conditions are better to initialize with the first interval and iterate starting from the second

Space: O(1), Time: O(NlogN)

# 4.01.2022
[2244. Minimum Rounds to Complete All Tasks](https://leetcode.com/problems/minimum-rounds-to-complete-all-tasks/description/) medium

[https://t.me/leetcode_daily_unstoppable/74](https://t.me/leetcode_daily_unstoppable/74)

[blog post](https://leetcode.com/problems/minimum-rounds-to-complete-all-tasks/solutions/2997205/kotlin-dfs-memo/)

```kotlin 
    fun minimumRounds(tasks: IntArray): Int {
        val counts = mutableMapOf<Int, Int>()
        tasks.forEach { counts[it] = 1 + counts.getOrDefault(it, 0)}
        var round = 0
        val cache = mutableMapOf<Int, Int>()
        fun fromCount(count: Int): Int {
            if (count == 0) return 0
            if (count < 0 || count == 1) return -1
            return if (count % 3 == 0) {
                count/3
            } else {
                cache.getOrPut(count, {
                    var v = fromCount(count - 3)
                    if (v == -1) v = fromCount(count - 2)
                    if (v == -1) -1 else 1 + v
                })
            }
        }
        counts.values.forEach { 
            val rounds = fromCount(it)
            if (rounds == -1) return -1
            round += rounds
        }
        return round
    }
```

For the optimal solution, we must take as many 3's of tasks as possible, then take 2's in any order.
First, we need to count how many tasks of each type there are. Next, we need to calculate the optimal `rounds` for the current tasks type count. There is a math solution, but ultimately we just can do DFS

Space: O(N), Time: O(N), counts range is always less than N

# 3.01.2022
[944. Delete Columns to Make Sorted](https://leetcode.com/problems/delete-columns-to-make-sorted/description/) easy

[https://t.me/leetcode_daily_unstoppable/73](https://t.me/leetcode_daily_unstoppable/73)

[blog post](https://leetcode.com/problems/delete-columns-to-make-sorted/solutions/2992229/kotlin-do-what-is-asked/)

```kotlin 
    fun minDeletionSize(strs: Array<String>): Int =
       (0..strs[0].lastIndex).asSequence().count { col ->
           (1..strs.lastIndex).asSequence().any { strs[it][col] < strs[it-1][col] }
        } 
```

Just do what is asked.
We can use Kotlin's `sequence` api.

Space: O(1), Time: O(wN)

# 2.01.2022
[520. Detect Capital](https://leetcode.com/problems/detect-capital/description/) easy

[https://t.me/leetcode_daily_unstoppable/72](https://t.me/leetcode_daily_unstoppable/72)

[blog post](https://leetcode.com/problems/detect-capital/solutions/2985088/kotlin-as-is/)

```kotlin 
    fun detectCapitalUse(word: String): Boolean =
       word.all { Character.isUpperCase(it) } ||
       word.all { Character.isLowerCase(it) } ||
       Character.isUpperCase(word[0]) && word.drop(1).all { Character.isLowerCase(it) }
```

We can do this optimally by checking the first character and then checking all the other characters in a single pass. Or we can write a more understandable code that directly translates from the problem description.
Let's write one-liner.

Space: O(1), Time: O(N)

# 1.01.2022
[290. Word Pattern](https://leetcode.com/problems/word-pattern/description/) easy

[https://t.me/leetcode_daily_unstoppable/71](https://t.me/leetcode_daily_unstoppable/71)

[blog post](https://leetcode.com/problems/word-pattern/solutions/2978765/kotlin-just-do/)

```kotlin 
    fun wordPattern(pattern: String, s: String): Boolean {
        val charToWord = Array<String>(27) { "" }
        val words = s.split(" ")
        if (words.size != pattern.length) return false
        words.forEachIndexed { i, w ->
            val cInd = pattern[i].toInt() - 'a'.toInt()

            if (charToWord[cInd] == "") {
                charToWord[cInd] = w
            } else if (charToWord[cInd] != w) return false
        }
        charToWord.sort()
        for (i in 1..26) 
            if (charToWord[i] != "" && charToWord[i] == charToWord[i-1]) 
                return false
        return true
    }
```

Each word must be in 1 to 1 relation with each character in the pattern. We can check this rule.

Use `string[27]` array for `char -> word` relation and also check each char have a unique word assigned.
* don't forget to check lengths

Space: O(N), Time: O(N)

# 31.12.2022
[980. Unique Paths III](https://leetcode.com/problems/unique-paths-iii/description/) hard

[https://t.me/leetcode_daily_unstoppable/69](https://t.me/leetcode_daily_unstoppable/69)

[blog post](https://leetcode.com/problems/unique-paths-iii/solutions/2974827/kotlin-dfs-backtracking/)

```kotlin 
    fun uniquePathsIII(grid: Array<IntArray>): Int {
        var countEmpty = 1
        var startY = 0
        var startX = 0
        for (y in 0..grid.lastIndex) {
            for (x in 0..grid[0].lastIndex) {
                when(grid[y][x]) {
                    0 -> countEmpty++
                    1 -> { startY = y; startX = x}
                    else -> Unit
                }
            }
        }
        fun dfs(y: Int, x: Int): Int {
            if (y < 0 || x < 0 || y >= grid.size || x >= grid[0].size) return 0
            val curr = grid[y][x]
            if (curr == 2) return if (countEmpty == 0) 1 else 0
            if (curr == -1) return 0
            grid[y][x] = -1
            countEmpty--
            val res =  dfs(y-1, x) + dfs(y, x-1) + dfs(y+1, x) + dfs(y, x+1)
            countEmpty++
            grid[y][x] = curr
            return res
        }
        return dfs(startY, startX)
    }
```

There is only `20x20` cells, we can brute-force the solution.
We can use DFS, and count how many empty cells passed. To avoid visiting cells twice, modify `grid` cell and then modify it back, like backtracking.

Space: O(1), Time: O(4^N)

# 30.12.2022
[797. All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target/description/) medium

[https://t.me/leetcode_daily_unstoppable/68](https://t.me/leetcode_daily_unstoppable/68)

[blog post](https://leetcode.com/problems/all-paths-from-source-to-target/solutions/1600383/kotlin-dfs-backtracking-java-iterative-dfs-stack/)

```kotlin 
    fun allPathsSourceTarget(graph: Array<IntArray>): List<List<Int>> {
        val res = mutableListOf<List<Int>>()
        val currPath = mutableListOf<Int>()
        fun dfs(curr: Int) {
            currPath += curr
            if (curr == graph.lastIndex) res += currPath.toList()
            graph[curr].forEach { dfs(it) }
            currPath.removeAt(currPath.lastIndex)
        }
        dfs(0)
        return res
    }
```
We must find all the paths, so there is no shortcuts to the visiting all of them.
One technique is backtracking - reuse existing visited list of nodes.

Space: O(VE), Time: O(VE)

# 29.12.2022
[1834. Single-Threaded CPU](https://leetcode.com/problems/single-threaded-cpu/description/) medium

[https://t.me/leetcode_daily_unstoppable/67](https://t.me/leetcode_daily_unstoppable/67)

[blog post](https://leetcode.com/problems/single-threaded-cpu/solutions/2966855/kotlin-two-heaps/)

```kotlin 
    fun getOrder(tasks: Array<IntArray>): IntArray {
        val pqSource = PriorityQueue<Int>(compareBy(
            { tasks[it][0] },
            { tasks[it][1] },
            { it }
        ))
        (0..tasks.lastIndex).forEach { pqSource.add(it) }
        val pq = PriorityQueue<Int>(compareBy(
            { tasks[it][1] },
            { it }
        ))
        val res = IntArray(tasks.size) { 0 }
        var time = 1 
        for(resPos in 0..tasks.lastIndex) {
            while (pqSource.isNotEmpty() && tasks[pqSource.peek()][0] <= time) {
                pq.add(pqSource.poll())
            }
            if (pq.isEmpty()) {
                //idle
                pq.add(pqSource.poll())
                time = tasks[pq.peek()][0]
            }
            //take task
            val taskInd = pq.poll()
            val task = tasks[taskInd]
            time += task[1]
            res[resPos] = taskInd
        }
        return res
    }
```
First we need to sort tasks by their availability (and other rules), 
then take tasks one by one and add them to another sorted set/heap where their start time doesn't matter, 
but running time and order does. When we take the task from the heap, we increase the time and fill in the heap.
* use two heaps, one for the source of tasks, another for the current available tasks.
* don't forget to increase time to the nearest task if all of them unavailable

Space: O(n), Time: O(nlogn)

# 28.12.2022
[1962. Remove Stones to Minimize the Total](https://leetcode.com/problems/remove-stones-to-minimize-the-total/description/) medium

[https://t.me/leetcode_daily_unstoppable/66](https://t.me/leetcode_daily_unstoppable/66)

[blog post](https://leetcode.com/problems/remove-stones-to-minimize-the-total/solutions/2961725/kotlin-priorityqueue/)

```kotlin 
    fun minStoneSum(piles: IntArray, k: Int): Int {
        val pq = PriorityQueue<Int>() 
        var sum = 0
        piles.forEach { 
            sum += it
            pq.add(-it) 
        }
        for (i in 1..k) {
            if (pq.isEmpty()) break
            val max = -pq.poll()
            if (max == 0) break 
            val newVal = Math.round(max/2.0).toInt()
            sum -= max - newVal
            pq.add(-newVal)
        }
        return sum
    }
```
By the problem definition, intuitively the best strategy is to reduce the maximum each time.
Use `PriorityQueue` to keep track of the maximum value and update it dynamically.
* one can use variable `sum` and update it each time.

Space: O(n), Time: O(nlogn)

# 27.12.2022
[2279. Maximum Bags With Full Capacity of Rocks](https://leetcode.com/problems/maximum-bags-with-full-capacity-of-rocks/description/) medium

[https://t.me/leetcode_daily_unstoppable/65](https://t.me/leetcode_daily_unstoppable/65)

[blog post](https://leetcode.com/problems/maximum-bags-with-full-capacity-of-rocks/solutions/2957374/kotlin-sort-greedy/)

```kotlin 
    fun maximumBags(capacity: IntArray, rocks: IntArray, additionalRocks: Int): Int {
       val inds = Array<Int>(capacity.size) { it }
       inds.sortWith(Comparator { a,b -> capacity[a]-rocks[a] - capacity[b] + rocks[b] })
       var rocksRemain = additionalRocks
       var countFull = 0
       for (i in 0..inds.lastIndex) {
           val toAdd = capacity[inds[i]] - rocks[inds[i]]
           if (toAdd > rocksRemain) break
           rocksRemain -= toAdd
           countFull++
       }
       return countFull
    }
```
We can logically deduce that the optimal solution is to take first bags with the smallest empty space.
Make an array of indexes and sort it by difference between `capacity` and `rocks`. Then just simulate rocks addition to each bug from the smallest empty space to the largest.

Space: O(n), Time: O(nlogn)

# 26.12.2022
[55. Jump Game](https://leetcode.com/problems/jump-game/description/) medium

[https://t.me/leetcode_daily_unstoppable/64](https://t.me/leetcode_daily_unstoppable/64)

[blog post](https://leetcode.com/problems/jump-game/solutions/2952687/kotlin-one-pass/)

```kotlin 
    fun canJump(nums: IntArray): Boolean {
       var minInd = nums.lastIndex 
       for (i in nums.lastIndex - 1 downTo 0) {
           if (nums[i] + i >= minInd) minInd = i
       }
       return minInd == 0
    }
```
For any position `i` we can reach the end if there is a `minInd` such that `nums[i] + i >= minInd` and `minInd` is a known to be reaching the end.
We can run from the end and update `minInd` - minimum index reaching the end.

Space: O(1), Time: O(N)

# 25.12.2022
[2389. Longest Subsequence With Limited Sum](https://leetcode.com/problems/longest-subsequence-with-limited-sum/description/) easy

[https://t.me/leetcode_daily_unstoppable/63](https://t.me/leetcode_daily_unstoppable/63)

[blog post](https://leetcode.com/problems/longest-subsequence-with-limited-sum/solutions/2948494/kotlin-sort-prefix-sum-binary-search/)

```kotlin 
    fun answerQueries(nums: IntArray, queries: IntArray): IntArray {
       nums.sort() 
       for (i in 1..nums.lastIndex) nums[i] += nums[i-1]
       return IntArray(queries.size) {
           val ind = nums.binarySearch(queries[it])
           if (ind < 0) -ind-1 else ind+1
       }
    }
```
We can logically deduce that for the maximum number of arguments we need to take as much as possible items from the smallest to the largest.
We can sort items. Then pre-compute `sums[i] = sum from [0..i]`. Then use binary search target sum in sums. Also, can modify `nums` but that's may be not necessary.

Space: O(N), Time: O(NlogN)

# 24.12.2022
[790. Domino and Tromino Tiling](https://leetcode.com/problems/domino-and-tromino-tiling/description/) medium

[https://t.me/leetcode_daily_unstoppable/62](https://t.me/leetcode_daily_unstoppable/62)

[blog post](https://leetcode.com/problems/domino-and-tromino-tiling/solutions/2946811/kotlin-dfs-memo/)

```kotlin 
    fun numTilings(n: Int): Int {
        val cache = Array<Array<Array<Long>>>(n) { Array(2) { Array(2) { -1L }}}
        fun dfs(pos: Int, topFree: Int, bottomFree: Int): Long {
            return when {
                pos > n -> 0L
                pos == n -> if (topFree==1 && bottomFree==1) 1L else 0L
                else -> {
                    var count = cache[pos][topFree][bottomFree]
                    if (count == -1L) {
                        count = 0L
                        when {
                            topFree==1 && bottomFree==1 -> {
                                count += dfs(pos+1, 1, 1) // vertical
                                count += dfs(pos+1, 0, 0) // horizontal
                                count += dfs(pos+1, 1, 0) // tromino top
                                count += dfs(pos+1, 0, 1) // tromino bottom
                            }
                            topFree==1 -> {
                                count += dfs(pos+1, 0, 0) // tromino
                                count += dfs(pos+1, 1, 0) // horizontal
                            }
                            bottomFree==1 -> {
                                count += dfs(pos+1, 0, 0) // tromino
                                count += dfs(pos+1, 0, 1) // horizontal
                            }
                        else -> {
                                count += dfs(pos+1, 1, 1) // skip
                            }
                        }

                        count = count % 1_000_000_007L
                    }
                    cache[pos][topFree][bottomFree] = count
                    count
                }
            }
        }
        return dfs(0, 1, 1).toInt()
    }
```

We can walk the board horizontally and monitor free cells. On each step, we can choose what figure to place. When end reached and there are no free cells, consider that a successful combination. Result depends only on the current position and on the top-bottom cell combination.* just do dfs+memo
* use array for a faster cache

Space: O(N), Time: O(N) - we only visit each column 3 times

# 23.12.2022
[309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/) medium

[https://t.me/leetcode_daily_unstoppable/61](https://t.me/leetcode_daily_unstoppable/61)

[blog post](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/solutions/1522780/java-0ms-from-recursion-with-memo-to-iterative-o-n-time-and-o-1-memory/)

```kotlin 
    data class K(val a:Int, val b: Boolean, val c:Boolean)
    fun maxProfit(prices: IntArray): Int {
        val cache = mutableMapOf<K, Int>()
        fun dfs(pos: Int, canSell: Boolean, canBuy: Boolean): Int {
            return if (pos == prices.size) 0
                else cache.getOrPut(K(pos, canSell, canBuy), {
                    val profitSkip = dfs(pos+1, canSell, !canSell)
                    val profitSell = if (canSell) {prices[pos] + dfs(pos+1, false, false)} else 0
                    val profitBuy = if (canBuy) {-prices[pos] + dfs(pos+1, true, false)} else 0
                    maxOf(profitSkip, profitBuy, profitSell)
                })
        }
        return dfs(0, false, true)
    }
```
Progress from dfs solution to memo. DFS solution - just choose what to do in this step, go next, then compare results and peek max.

Space: O(N), Time: O(N)


# 22.12.2022
[834. Sum of Distances in Tree](https://leetcode.com/problems/sum-of-distances-in-tree/description/) hard

[https://t.me/leetcode_daily_unstoppable/60](https://t.me/leetcode_daily_unstoppable/60)

[blog post](https://leetcode.com/problems/sum-of-distances-in-tree/solutions/1443979/kotlin-java-2-dfs-diagramm-to-invent-the-change-root-equation/)

```kotlin 
    fun sumOfDistancesInTree(n: Int, edges: Array<IntArray>): IntArray {
        val graph = mutableMapOf<Int, MutableList<Int>>()
        edges.forEach { (from, to) -> 
            graph.getOrPut(from, { mutableListOf() }) += to
            graph.getOrPut(to, { mutableListOf() }) += from
        }
        val counts = IntArray(n) { 1 }
        val sums = IntArray(n) { 0 }
        fun distSum(pos: Int, visited: Int) {
            graph[pos]?.forEach {
                if (it != visited) {
                    distSum(it, pos)
                    counts[pos] += counts[it]
                    sums[pos] += counts[it] + sums[it]
                }
            }
        }
        fun dfs(pos: Int, visited: Int) {
            graph[pos]?.forEach {
                if (it != visited) {
                    sums[it] = sums[pos] - counts[it] + (n - counts[it])
                    dfs(it, pos)
                }
            }
        }
        distSum(0, -1)
        dfs(0, -1)
        return sums
    }
```
We can do the job for item #0, then we need to invent a formula to reuse some data when we change the node.

How to mathematically prove formula for a new sum:
![image](https://assets.leetcode.com/users/images/f7d1ffbc-7761-4cff-a219-58e1a433bd1c_1630765686.6135957.png)

![image.png](https://assets.leetcode.com/users/images/b2c81eba-e532-43cc-ae6a-6aec3eed57f9_1671730095.0767915.png)
Store count of children in a `counts` array, and sum of the distances to children in a `dist` array. In a first DFS traverse from a node 0 and fill the arrays. In a second DFS only modify `dist` based on previous computed `dist` value, using formula: `sum[curr] = sum[prev] - count[curr] + (N - count[curr])`

Space: O(N), Time: O(N)

# 21.12.2022
[886. Possible Bipartition](https://leetcode.com/problems/possible-bipartition/description/) medium

[https://t.me/leetcode_daily_unstoppable/59](https://t.me/leetcode_daily_unstoppable/59)

[blog post](https://leetcode.com/problems/possible-bipartition/solutions/2936306/kotlin-union-find/)

```kotlin 
fun possibleBipartition(n: Int, dislikes: Array<IntArray>): Boolean {
	val love = IntArray(n+1) { it }
	fun leader(x: Int): Int {
		var i = x
		while (love[i] != i) i = love[i]
		love[x] = i
		return i
	}
	val hate = IntArray(n+1) { -1 }
	dislikes.forEach { (one, two) ->
		val leaderOne = leader(one)
		val leaderTwo = leader(two)
		val enemyOfOne = hate[leaderOne]
		val enemyOfTwo = hate[leaderTwo]
		if (enemyOfOne != -1 && enemyOfOne == enemyOfTwo) return false
		if (enemyOfOne != -1) {
			love[leader(enemyOfOne)] = leaderTwo
		}
		if (enemyOfTwo != -1) {
			love[leader(enemyOfTwo)] = leaderOne
		}
		hate[leaderOne] = leaderTwo
		hate[leaderTwo] = leaderOne
	}
	return true
}
```
We need somehow to union people that hate the same people. We can do it making someone a leader of a group and make just leaders to hate each other.

Keep track of the leaders hating each other in the `hate` array, and people loving their leader in `love` array. (`love` array is basically a Union-Find).
* also use path compression for `leader` method

Space: O(N), Time: O(N) - adding to Union-Find is O(1) amortised

# 20.12.2022
[841. Keys and Rooms](https://leetcode.com/problems/keys-and-rooms/description/) medium

[https://t.me/leetcode_daily_unstoppable/58](https://t.me/leetcode_daily_unstoppable/58)

[blog post](https://leetcode.com/problems/keys-and-rooms/solutions/2932740/kotlin-dfs/)

```kotlin 
    fun canVisitAllRooms(rooms: List<List<Int>>): Boolean {
       val visited = hashSetOf(0)
       with(ArrayDeque<Int>()) {
           add(0)
           while(isNotEmpty()) {
               rooms[poll()].forEach {
                   if (visited.add(it)) add(it)
               }
           }
       }
       return visited.size == rooms.size
    }
```
We need to visit each room, and we have positions of the other rooms and a start position. This is a DFS problem.
Keep all visited rooms numbers in a hash set and check the final size. Other solution is to use boolean array and a counter of the visited rooms.

Space: O(N) - for queue and visited set, Time: O(N) - visit all the rooms once

# 19.12.2022
[1971. Find if Path Exists in Graph](https://leetcode.com/problems/find-if-path-exists-in-graph/description/) easy

[https://t.me/leetcode_daily_unstoppable/57](https://t.me/leetcode_daily_unstoppable/57)

[blog post](https://leetcode.com/problems/find-if-path-exists-in-graph/solutions/2928882/kotlin-bfs/)

```kotlin 
    fun validPath(n: Int, edges: Array<IntArray>, source: Int, destination: Int): Boolean {
        if (source == destination) return true
        val graph = mutableMapOf<Int, MutableList<Int>>()
        edges.forEach { (from, to) -> 
            graph.getOrPut(from, { mutableListOf() }).add(to)
            graph.getOrPut(to, { mutableListOf() }).add(from)
        }
        val visited = mutableSetOf<Int>()
        with(ArrayDeque<Int>()) {
            add(source)
            var depth = 0
            while(isNotEmpty() && ++depth < n) {
                repeat(size) {
                    graph[poll()]?.forEach {
                        if (it == destination) return true
                        if (visited.add(it)) add(it) 
                    }
                }
            }
        }
        return false
    }
```

BFS will do the job.
Make node to nodes map, keep visited set and use queue for BFS.
* also path can't be longer than n elements

Space: O(N), Time: O(N)

# 18.12.2022
[739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/description/) medium

[https://t.me/leetcode_daily_unstoppable/55](https://t.me/leetcode_daily_unstoppable/55)

[blog post](https://leetcode.com/problems/daily-temperatures/solutions/2924323/kotlin-increasing-stack/)

```kotlin 
    fun dailyTemperatures(temperatures: IntArray): IntArray {
       val stack = Stack<Int>() 
       val res = IntArray(temperatures.size) { 0 }
       for (i in temperatures.lastIndex downTo 0) {
           while(stack.isNotEmpty() && temperatures[stack.peek()] <= temperatures[i]) stack.pop()
           if (stack.isNotEmpty()) {
               res[i] = stack.peek() - i
           }
           stack.push(i)
       }
       return res
    }
```

Intuitively, we want to go from the end of the array to the start and keep the maximum value. But, that doesn't work, because we must also store smaller numbers, as they are closer in distance.
For example, `4 3 5 6`, when we observe `4` we must compare it to `5`, not to `6`. So, we store not just max, but increasing max: `3 5 6`, and throw away all numbers smaller than current, `3 < 4` - pop().

We will iterate in reverse order, storing indexes in increasing by temperatures stack.

Space: O(N), Time: O(N)

# 17.12.2022
[150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/description/) medium

[https://t.me/leetcode_daily_unstoppable/54](https://t.me/leetcode_daily_unstoppable/54)

[blog post](https://leetcode.com/problems/evaluate-reverse-polish-notation/solutions/2922482/kotlin-stack/)

```kotlin 
    fun evalRPN(tokens: Array<String>): Int = with(Stack<Int>()) {
        tokens.forEach {
            when(it) {
                "+" -> push(pop() + pop())
                "-" -> push(-pop() + pop())
                "*" -> push(pop() * pop())
                "/" -> with(pop()) { push(pop()/this) }
                else -> push(it.toInt())
            }
        }
        pop()
    }
```

Reverse polish notations made explicitly for calculation using stack. Just execute every operation immediately using last two numbers in the stack and push the result.
* be aware of the order of the operands

Space: O(N), Time: O(N)

# 16.12.2022
[232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/description/) easy

[https://t.me/leetcode_daily_unstoppable/53](https://t.me/leetcode_daily_unstoppable/53)

[blog post](https://leetcode.com/problems/implement-queue-using-stacks/solutions/2918693/kotlin-head-tail/)

```kotlin 
class MyQueue() {
	val head = Stack<Int>()
	val tail = Stack<Int>()

	//  []       []
	//  1 2 3 4 -> 4 3 2 - 1
	//  5         4 3 2
	//            4 3 2 5
	fun push(x: Int) {
		head.push(x)
	}

	fun pop(): Int {
		peek()

		return tail.pop()
	}

	fun peek(): Int {
		if (tail.isEmpty()) while(head.isNotEmpty()) tail.push(head.pop())

		return tail.peek()
	}

	fun empty(): Boolean = head.isEmpty() && tail.isEmpty()

}
```
One stack for the head of the queue and other for the tail.
When we need to do `pop` we first drain from one stack to another, so items order will be restored.
* we can skip rotation on push if we fill tail only when its empty

Space: O(1), Time: O(1)

# 15.12.2022
[1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/description/) medium

[https://t.me/leetcode_daily_unstoppable/52](https://t.me/leetcode_daily_unstoppable/52)

[blog post](https://leetcode.com/problems/longest-common-subsequence/solutions/2915134/kotlin-dfs-memo/)

```kotlin 
    fun longestCommonSubsequence(text1: String, text2: String): Int {
        val cache = Array(text1.length + 1) { IntArray(text2.length + 1) { -1 } }
        fun dfs(pos1: Int, pos2: Int): Int {
            if (pos1 == text1.length) return 0
            if (pos2 == text2.length) return 0
            val c1 = text1[pos1]
            val c2 = text2[pos2]
            if (cache[pos1][pos2] != -1) return cache[pos1][pos2]
            val res = if (c1 == c2) {
                    1 + dfs(pos1 + 1, pos2 + 1)
                } else {
                    maxOf(dfs(pos1, pos2+1), dfs(pos1+1, pos2))
                }
            cache[pos1][pos2] = res
            return res
        }
        return dfs(0, 0)
    }
```
We can walk the two strings simultaneously and compare their chars. If they are the same, the optimal way will be to use those chars and continue exploring next. If they are not, we have two choices: use the first char and skip the second or skip the first but use the second.
Also, observing our algorithm we see, the result so far is only dependent of the positions from which we begin to search (and all the remaining characters). And also see that the calls are repetitive. That mean we can cache the result. (meaning this is a dynamic programming solution).
Use depth first search by starting positions and memoize results in a two dimension array. Another approach will be bottom up iteration and filling the same array.

Space: O(N^2), Time: O(N^2)

# 14.12.2022
[198. House Robber](https://leetcode.com/problems/house-robber/description/) medium

[https://t.me/leetcode_daily_unstoppable/51](https://t.me/leetcode_daily_unstoppable/51)

[blog post](https://leetcode.com/problems/house-robber/solutions/2911816/kotlin-dfs-memo/)

```kotlin 
    fun rob(nums: IntArray): Int {
        val cache = mutableMapOf<Int, Int>()
        fun dfs(pos: Int): Int {
            if (pos > nums.lastIndex) return 0
            return cache.getOrPut(pos) {
                maxOf(nums[pos] + dfs(pos+2), dfs(pos+1))
            }
        } 
        return dfs(0)
    }
```

Exploring each house one by one we can make a decision to rob or not to rob.
The result is only depends on our current position (and all houses that are remaining to rob) and decision, so we can memoize it based on position.

We can use memoization or walk houses bottom up.

Space: O(N), Time: O(N)

# 13.12.2022
[931. Minimum Falling Path Sum](https://leetcode.com/problems/minimum-falling-path-sum/description/) medium

[https://t.me/leetcode_daily_unstoppable/50](https://t.me/leetcode_daily_unstoppable/50)

[blog post](https://leetcode.com/problems/minimum-falling-path-sum/solutions/2908108/kotlin-running-sum/)

```kotlin 
    fun minFallingPathSum(matrix: Array<IntArray>): Int {
       for (y in matrix.lastIndex-1 downTo 0) {
           val currRow = matrix[y]
           val nextRow = matrix[y+1]
           for (x in 0..matrix[0].lastIndex) {
               val left = if (x > 0) nextRow[x-1] else Int.MAX_VALUE
               val bottom = nextRow[x]
               val right = if (x < matrix[0].lastIndex) nextRow[x+1] else Int.MAX_VALUE
               val minSum = currRow[x] + minOf(left, bottom, right)
               currRow[x] = minSum
           }
       } 
       return matrix[0].min()!!
    }
```
There is only three ways from any cell to it's siblings. We can compute all three paths sums for all cells in a row so far. And then choose the smallest.
Iterate over rows and compute prefix sums of current + minOf(left min sum, bottom min sum, right min sum)

Space: O(N), Time: O(N^2)

# 12.12.2022
[70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/description/) easy

[https://t.me/leetcode_daily_unstoppable/49](https://t.me/leetcode_daily_unstoppable/49)

[blog post](https://leetcode.com/problems/climbing-stairs/solutions/2904774/kotlin-dfs-memo/)

```kotlin 
    val cache = mutableMapOf<Int, Int>()
    fun climbStairs(n: Int): Int = when {
        n < 1  -> 0
        n == 1 -> 1
        n == 2 -> 2
        else -> cache.getOrPut(n) {
            climbStairs(n-1) + climbStairs(n-2)
        }
    }
```
You can observe that result is only depend on input n. And also that result(n) = result(n-1) + result(n-2).
Just use memoization for storing already solved inputs.

Space: O(N), Time: O(N)

# 11.12.2022
[124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/description/) hard

[https://t.me/leetcode_daily_unstoppable/48](https://t.me/leetcode_daily_unstoppable/48)

[blog post](https://leetcode.com/problems/binary-tree-maximum-path-sum/solutions/2900498/kotlin-very-bad-problem-definition/)

```kotlin 
    fun maxPathSum(root: TreeNode?): Int {
        fun dfs(root: TreeNode): Pair<Int, Int> {
            val lt = root.left
            val rt = root.right
            if (lt == null && rt == null) return root.`val` to root.`val`
            if (lt == null || rt == null) {
                val sub = dfs(if (lt == null) rt else lt)
                val currRes = root.`val` + sub.second
                val maxRes = maxOf(sub.first, currRes, root.`val`)
                val maxPath = maxOf(root.`val`, root.`val` + sub.second)
                return maxRes to maxPath
            } else {
                val left = dfs(root.left)
                val right = dfs(root.right)
                val currRes1 = root.`val` + left.second + right.second
                val currRes2 = root.`val`
                val currRes3 = root.`val` + left.second
                val currRes4 = root.`val` + right.second
                val max1 = maxOf(currRes1, currRes2)
                val max2 = maxOf(currRes3, currRes4)
                val maxRes = maxOf(left.first, right.first, maxOf(max1, max2))
                val maxPath = maxOf(root.`val`, root.`val` + maxOf(left.second, right.second))
                return maxRes to maxPath
            }
        }
        return if (root == null) 0 else dfs(root).first
    }
```

Space: O(logN), Time: O(N)

# 10.12.2022
[1339. Maximum Product of Splitted Binary Tree](https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/description/) medium

[https://t.me/leetcode_daily_unstoppable/47](https://t.me/leetcode_daily_unstoppable/47)

[blog post](https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/solutions/2896607/kotlin-two-dfs/)

```kotlin
    fun maxProduct(root: TreeNode?): Int {
        fun sumDfs(root: TreeNode?): Long {
            return if (root == null) 0L
            else with(root) { `val`.toLong() + sumDfs(left) + sumDfs(right) }
        }
        val total = sumDfs(root)
        fun dfs(root: TreeNode?) : Pair<Long, Long> {
            if (root == null) return Pair(0,0)
            val left = dfs(root.left)
            val right = dfs(root.right)
            val sum = left.first + root.`val`.toLong() + right.first
            val productLeft = left.first * (total - left.first) 
            val productRight = right.first * (total - right.first)
            val prevProductMax = maxOf(right.second, left.second)
            return sum to maxOf(productLeft, productRight, prevProductMax)
        }
        return (dfs(root).second % 1_000_000_007L).toInt()
    }
```
Just iterate over all items and compute all products.
We need to compute total sum before making the main traversal.

Space: O(logN), Time: O(N)

# 9.12.2022
[1026. Maximum Difference Between Node and Ancestor](https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/description/) medium

[https://t.me/leetcode_daily_unstoppable/46](https://t.me/leetcode_daily_unstoppable/46)

[blog post](https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/solutions/2894948/kotlin-dfs/)

```kotlin
    fun maxAncestorDiff(root: TreeNode?): Int {
        root?: return 0

        fun dfs(root: TreeNode, min: Int = root.`val`, max: Int = root.`val`): Int {
            val v = root.`val`
            val currDiff = maxOf(Math.abs(v - min), Math.abs(v - max))
            val currMin = minOf(min, v)
            val currMax = maxOf(max, v)
            val leftDiff = root.left?.let { dfs(it, currMin, currMax) } ?: 0
            val rightDiff = root.right?.let { dfs(it, currMin, currMax) } ?: 0
            return maxOf(currDiff, leftDiff, rightDiff)
        }
        
        return dfs(root)
    }
```
Based on math we can assume, that max difference is one of the two: (curr - max so far) or (curr - min so far).
Like, for example, let our curr value be `3`, and from all visited we have min `0` and max `7`.
```
 0--3---7
```
* we can write helper recoursive method and compute max and min so far

Space: O(logN), Time: O(N)

# 8.12.2022
[872. Leaf-Similar Trees](https://leetcode.com/problems/leaf-similar-trees/solutions/) easy

[https://t.me/leetcode_daily_unstoppable/45](https://t.me/leetcode_daily_unstoppable/45)

```kotlin
    fun leafSimilar(root1: TreeNode?, root2: TreeNode?): Boolean {
        fun dfs(root: TreeNode?): List<Int> {
            return when {
                root == null -> listOf()
                root.left == null && root.right == null -> listOf(root.`val`)
                else -> dfs(root.left) + dfs(root.right)
            }
        }
        
        return dfs(root1) == dfs(root2)
    }
```
There is only 200 items, so we can concatenate lists.
One optimization would be to collect only first tree and just compare it to the second tree while doing the inorder traverse.

Space: O(N), Time: O(N)

# 7.12.2022
[938. Range Sum of BST](https://leetcode.com/problems/range-sum-of-bst/description/) easy

[https://t.me/leetcode_daily_unstoppable/44](https://t.me/leetcode_daily_unstoppable/44)

```kotlin
    fun rangeSumBST(root: TreeNode?, low: Int, high: Int): Int =
	if (root == null) 0 else
		with(root) {
			(if (`val` in low..high) `val` else 0) +
				(if (`val` < low) 0 else rangeSumBST(left, low, high)) +
				(if (`val` > high) 0 else rangeSumBST(right, low, high))
		}
```
* be careful with ternary operations, better wrap them in a brackets

Space: O(log N), Time: O(R), r - is a range [low, high]

# 6.12.2022
[328. Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/description/) medium

[https://t.me/leetcode_daily_unstoppable/43](https://t.me/leetcode_daily_unstoppable/43)

```kotlin
       // 1 2
    fun oddEvenList(head: ListNode?): ListNode? {
       var odd = head //1
       val evenHead = head?.next
       var even = head?.next //2
       while(even!=null) { //2
           val oddNext = odd?.next?.next //null
           val evenNext = even?.next?.next //null
           odd?.next = oddNext // 1->null
           even?.next = evenNext //2->null
           if (oddNext != null) odd = oddNext //
           even = evenNext // null
       }
       odd?.next = evenHead // 1->2
       return head //1->2->null
    }
```
* be careful and store evenHead in a separate variable

Space: O(1), Time: O(n)


# 5.12.2022
[876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/) easy

[https://t.me/leetcode_daily_unstoppable/42](https://t.me/leetcode_daily_unstoppable/42)

```kotlin
  fun middleNode(head: ListNode?, fast: ListNode? = head): ListNode? =
        if (fast?.next == null) head else middleNode(head?.next, fast?.next?.next)
```
* one-liner, but in the interview (or production) I would prefer to write a loop

Space: O(n), Time: O(n)

# 4.12.2022
[2256. Minimum Average Difference](https://leetcode.com/problems/minimum-average-difference/) medium

[https://t.me/leetcode_daily_unstoppable/41](https://t.me/leetcode_daily_unstoppable/41)

```kotlin
    fun minimumAverageDifference(nums: IntArray): Int {
        var sum = 0L
        nums.forEach { sum += it.toLong() }
        var leftSum = 0L
        var min = Long.MAX_VALUE
        var minInd = 0
        for (i in 0..nums.lastIndex) {
            val leftCount = (i+1).toLong()
            leftSum += nums[i].toLong()
            val front = leftSum/leftCount
            val rightCount = nums.size.toLong() - leftCount
            val rightSum = sum - leftSum
            val back = if (rightCount == 0L) 0L else rightSum/rightCount
            val diff = Math.abs(front - back)
            if (diff < min) {
                min = diff
                minInd = i
            }
        }
        return minInd
    }
```
### Intuition

Two pointers, one for even, one for odd indexes.
### Approach

To avoid mistakes you need to be verbose, and don't skip operations:
* store evenHead in a separate variable
* don't switch links before both pointers jumped
* don't make odd pointer null
* try to run for simple input `1->2->null` by yourself

  
Space: O(1), Time: O(n)

# 3.12.2022
[451. Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/) medium

[https://t.me/leetcode_daily_unstoppable/40](https://t.me/leetcode_daily_unstoppable/40)

```kotlin
    fun frequencySort(s: String): String =
        s.groupBy { it }
        .values
        .map { it to it.size }
        .sortedBy { -it.second }
        .map { it.first }
        .flatten()
        .joinToString("")
```
Very simple task, can be written in a functional style.
Space: O(n), Time: O(n)

# 2.12.2022
[https://leetcode.com/problems/determine-if-two-strings-are-close/](https://leetcode.com/problems/determine-if-two-strings-are-close/) medium

[https://t.me/leetcode_daily_unstoppable/39](https://t.me/leetcode_daily_unstoppable/39)

```kotlin
    // cabbba -> c aa bbb -> 1 2 3 
    // a bb ccc -> 1 2 3
    // uau
    // ssx
    fun closeStrings(word1: String, word2: String, 
         f: (String) -> List<Int> = { it.groupBy { it }.values.map { it.size }.sorted() }
    ): Boolean = f(word1) == f(word2) && word1.toSet() == word2.toSet()
```
That is a simple task, you just need to know what exactly you asked for.
Space: O(n), Time: O(n)

# 1.12.2022
[1704. Determine if String Halves Are Alike](https://leetcode.com/problems/determine-if-string-halves-are-alike/) easy

[https://t.me/leetcode_daily_unstoppable/38](https://t.me/leetcode_daily_unstoppable/38)

```kotlin
    fun halvesAreAlike(s: String): Boolean {
        val vowels = setOf('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
        var c1 = 0
        var c2 = 0
        s.forEachIndexed { i, c -> 
            if (c in vowels) {
                if (i < s.length / 2) c1++ else c2++
            }
        }
        return c1 == c2
    }
```
Just do what is asked.

O(N) time, O(1) space

# 30.11.2022
[1207. Unique Number of Occurrences](https://leetcode.com/problems/unique-number-of-occurrences/) easy

[https://t.me/leetcode_daily_unstoppable/36](https://t.me/leetcode_daily_unstoppable/36)

```kotlin
fun uniqueOccurrences(arr: IntArray): Boolean {
	val counter = mutableMapOf<Int, Int>()
	arr.forEach { n -> counter[n] = 1 + (counter[n] ?: 0) }
	val freq = mutableSetOf<Int>()
	return !counter.values.any { count -> !freq.add(count) }
}
```
Nothing interesting, just count and filter.

O(N) time, O(N) space

# 29.11.2022
[380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/) medium

[https://t.me/leetcode_daily_unstoppable/35](https://t.me/leetcode_daily_unstoppable/35)

```kotlin
class RandomizedSet() {
    val rnd = Random(0)
    val list = mutableListOf<Int>()
    val vToInd = mutableMapOf<Int, Int?>()
    fun insert(v: Int): Boolean {
        if (!vToInd.contains(v)) {
            vToInd[v] = list.size
            list.add(v)
            return true
        }
        return false
    }
    fun remove(v: Int): Boolean {
        val ind = vToInd[v] ?: return false
        val prevLast = list[list.lastIndex]
        list[ind] = prevLast
        vToInd[prevLast] = ind
        list.removeAt(list.lastIndex)
        vToInd.remove(v)
        return true
    }
    fun getRandom(): Int = list[rnd.nextInt(list.size)]
}

```
The task is simple, one trick is to remove elements from the end of the list, and replacing item with the last one.
Some thoughts:
* don't optimize lines of code, that can backfire. You can use syntax sugar, clever operations inlining, but also can shoot in the foot.

O(1) time, O(N) space

# 28.11.2022
[2225. Find Players With Zero or One Losses](https://leetcode.com/problems/find-players-with-zero-or-one-losses/) medium

[https://t.me/leetcode_daily_unstoppable/34](https://t.me/leetcode_daily_unstoppable/34)

```kotlin
    fun findWinners(matches: Array<IntArray>): List<List<Int>> {
        val winners = mutableMapOf<Int, Int>()
        val losers = mutableMapOf<Int, Int>()
        matches.forEach { (w, l) ->
            winners[w] = 1 + (winners[w]?:0)
            losers[l] = 1 + (losers[l]?:0)
        }
        return listOf(
            winners.keys
                .filter { !losers.contains(it) }
                .sorted(),
            losers
                .filter { (k, v) -> v == 1 }
                .map { (k, v) -> k}
                .sorted()
        )
    }
```

Just do what is asked.

O(NlogN) time, O(N) space

# 27.11.2022
[446. Arithmetic Slices II - Subsequence](https://leetcode.com/problems/arithmetic-slices-ii-subsequence/) hard

[https://t.me/leetcode_daily_unstoppable/33](https://t.me/leetcode_daily_unstoppable/33)

```kotlin
    fun numberOfArithmeticSlices(nums: IntArray): Int {
        // 0 1 2 3 4 5 
        // 1 2 3 1 2 3                diff = 1
        //   ^     ^ *                dp[5][diff] = 
        //   |     |  \__ curr        1 + dp[1][diff] +
        //  prev   |                  1 + dp[4][diff]
        //        prev
        // 
        val dp = Array(nums.size) { mutableMapOf<Long, Long> () }
        for (curr in 0..nums.lastIndex) {
            for (prev in 0 until curr) {
                val diff = nums[curr].toLong() - nums[prev].toLong()
                dp[curr][diff] = 1 + (dp[curr][diff]?:0L) + (dp[prev][diff]?:0L)
            }
        }
        return dp.map { it.values.sum()!! }.sum().toInt() - (nums.size)*(nums.size-1)/2
    }
```

dp[i][d] is the number of subsequences in range [0..i] with difference = d
```kotlin
array: "1 2 3 1 2 3"
For items  1  2  curr = 2:
diff = 1,  dp = 1
For items  1  2  3  curr = 3:
diff = 2,  dp = 1
diff = 1,  dp = 2
For items  1  2  3  1  curr = 1:
diff = 0,  dp = 1
diff = -1,  dp = 1
diff = -2,  dp = 1
For items  1  2  3  1  2  curr = 2:
diff = 1,  dp = 2
diff = 0,  dp = 1
diff = -1,  dp = 1
For items  1  2  3  1  2  3  curr = 3:
diff = 2,  dp = 2
diff = 1,  dp = 5
diff = 0,  dp = 1
```
and finally, we need to subtract all the sequences of length 2 and 1,
count of them is (n)*(n-1)/2

O(N^2) time, O(N^2) space

# 26.11.2022
[1235. Maximum Profit in Job Scheduling](https://leetcode.com/problems/maximum-profit-in-job-scheduling/) hard

[https://t.me/leetcode_daily_unstoppable/32](https://t.me/leetcode_daily_unstoppable/32)

```kotlin
    fun jobScheduling(startTime: IntArray, endTime: IntArray, profit: IntArray): Int {
        val n = startTime.size
        val inds = Array<Int>(n) { it }
        inds.sortWith (Comparator<Int> { a, b -> 
            if (startTime[a] == startTime[b])
                endTime[a] - endTime[b]
            else
                startTime[a] - startTime[b]
        })
        val maxProfit = IntArray(n) { 0 }
        maxProfit[n-1] = profit[inds[n-1]]
        for (i in n-2 downTo 0) {
            val ind = inds[i]
            val end = endTime[ind]
            val prof = profit[ind]
            
            var lo = l + 1
            var hi = n - 1
            var nonOverlapProfit = 0
            while (lo <= hi) {
                val mid = lo + (hi - lo) / 2
                if (end <= startTime[inds[mid]]) {
                    nonOverlapProfit = maxOf(nonOverlapProfit, maxProfit[mid])
                    hi = mid - 1
                } else lo = mid + 1
            }
            maxProfit[i] = maxOf(prof + nonOverlapProfit, maxProfit[i+1])
        }
        return maxProfit[0]
    }
```

Use the hints from the description.
THis cannot be solved greedily, because you need to find next non-overlapping job.
Dynamic programming equation: from last job to the current, result is max of next result and current + next non-overlapping result.
```
f(i) = max(f(i+1), profit[i] + f(j)), where j is the first non-overlapping job after i.
```
Also, instead of linear search for non overlapping job, use binary search.

O(NlogN) time, O(N) space

# 25.11.2022
[907. Sum of Subarray Minimums](https://leetcode.com/problems/sum-of-subarray-minimums/) medium

```kotlin
    data class V(val v: Int, val count: Int)
    fun sumSubarrayMins(arr: IntArray): Int {
        val M = 1_000_000_007
        // 1 2 3 4 2 2 3 4
        //  1 2 3 2 2 2 3
        //   1 2 2 2 2 2
        //    1 2 2 2 2
        //     1 2 2 2
        //      1 2 2
        //       1 2
        //        1
        // f(1) = 1
        // f(2) = 2>1 ? f(1) + [1, 2]
        // f(3) = 3>2 ? f(2) + [1, 2, 3]
        // f(4) = 4>3 ? f(3) + [1, 2, 3, 4]
        // f(2) = 2<4 ? f(4) + [1, 2, 2, 2, 2] (1, 2, 3, 4 -> 3-2, 4-2, +2)
        // f(2) = 2=2 ? f(2) + [1, 2, 2, 2, 2, 2]
        // f(3) = 3>2 ? f(2) + [1, 2, 2, 2, 2, 2, 3]
        // f(4) = 4>3 ? f(3) + [1, 2, 2, 2, 2, 2, 3, 4]
        // 3 1 2 4    f(3) = 3    sum = 3  stack: [3]
        //  1 1 2     f(1): 3 > 1 , remove V(3,1), sum = sum - 3 + 1*2= 2, f=3+2=5, [(1,2)]
        //   1 1      f(2): 2>1, sum += 2 = 4, f+=4=9
        //    1       f(4): 4>2, sum+=4=8, f+=8=17
        val stack = Stack<V>()
        var f = 0
        var sum = 0
        arr.forEach { n ->
            var countRemoved = 0
            while (stack.isNotEmpty() && stack.peek().v > n) {
                val v = stack.pop()
                countRemoved += v.count
                var removedSum = (v.v*v.count) % M
                if (removedSum < 0) removedSum = M + removedSum
                sum = (sum - removedSum) % M
                if (sum < 0) sum = sum + M
            }
            val count = countRemoved + 1
            stack.add(V(n, count))
            sum = (sum + (n * count) % M) % M
            f = (f + sum) % M
            
        }
        return f
    }
```

First attempt is to build an N^2 tree of minimums, comparing adjacent elements row by row and finding a minimum.
That will take O(N^2) time and gives TLE. 
Next observe that there is a repetition of the results if we computing result for each new element:
result = previous result + some new elements.
That new elements are also have a law of repetition: 
sum = current element + if (current element < previous element) count of previous elements * current element else previous sum
We can use a stack to keep lowest previous elements, all values in stack must be less than current element.

O(N) time, O(N) space

# 24.11.2022
[79. Word Search](https://leetcode.com/problems/word-search/) medium

```kotlin
    fun exist(board: Array<CharArray>, word: String): Boolean {
        fun dfs(y: Int, x: Int, pos: Int): Boolean {
            if (pos == word.length) return true
            if (y < 0 || x < 0 || y == board.size || x == board[0].size) return false
            val c = board[y][x]
            if (c != word[pos]) return false
            board[y][x] = '.'
            val res = dfs(y-1, x, pos+1)
                   || dfs(y+1, x, pos+1)
                   || dfs(y, x-1, pos+1)
                   || dfs(y, x+1, pos+1)
            board[y][x] = c
            return res
        }
        for (y in 0..board.lastIndex) {
            for (x in 0..board[0].lastIndex) {
                if (dfs(y, x, 0)) return true
            }
        }
        return false
    }
```

We can brute force this problem. Backtracking help to preserve memory.

Complexity: O(M*N*W)
Memory: O(W)

# 23.11.2022
[https://leetcode.com/problems/valid-sudoku/](https://leetcode.com/problems/valid-sudoku/) medium

```kotlin
    fun isValidSudoku(board: Array<CharArray>): Boolean {
        val cell9 = arrayOf(0 to 0, 0 to 1, 0 to 2, 
                            1 to 0, 1 to 1, 1 to 2, 
                            2 to 0, 2 to 1, 2 to 2)
        val starts = arrayOf(0 to 0, 0 to 3, 0 to 6, 
                             3 to 0, 3 to 3, 3 to 6, 
                             6 to 0, 6 to 3, 6 to 6)
        return !starts.any { (sy, sx) ->
                val visited = HashSet<Char>()
                cell9.any { (dy, dx) ->
                    val c = board[sy+dy][sx+dx]
                    c != '.' && !visited.add(c)
                }
            } && !board.any { row -> 
                val visited = HashSet<Char>()
                row.any { it != '.' && !visited.add(it) }
            } && !(0..8).any { x ->
                val visited = HashSet<Char>()
                (0..8).any { board[it][x] != '.' && !visited.add(board[it][x]) }
            }
    }
```

This is an easy problem, just do what is asked.

Complexity: O(N)
Memory: O(N), N = 81, so it O(1)

# 22.11.2022
[https://leetcode.com/problems/perfect-squares/](https://leetcode.com/problems/perfect-squares/) medium

```kotlin
    val cache = mutableMapOf<Int, Int>()
    fun numSquares(n: Int): Int {
        if (n < 0) return -1
        if (n == 0) return 0
        if (cache[n] != null) return cache[n]!!
        var min = Int.MAX_VALUE
        for (x in Math.sqrt(n.toDouble()).toInt() downTo 1) {
            val res = numSquares(n - x*x)
            if (res != -1) {
                min = minOf(min, 1 + res)
            }
        }
        if (min == Int.MAX_VALUE) min = -1
        cache[n] = min
        return min
    }
```

The problem gives stable answers for any argument n. 
So, we can use memoization technique and search from the biggest square to the smallest one.

Complexity: O(Nsqrt(N))
Memory: O(N)

# 21.11.2022
[https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/](https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/) medium

```kotlin
    fun nearestExit(maze: Array<CharArray>, entrance: IntArray): Int {
        val queue = ArrayDeque<Pair<Int, Int>>()
        queue.add(entrance[1] to entrance[0])
        maze[entrance[0]][entrance[1]] = 'x'
        var steps = 1
        val directions = intArrayOf(-1, 0, 1, 0, -1)
        while(queue.isNotEmpty()) {
            repeat(queue.size){
                val (x, y) = queue.poll()
                for (i in 1..directions.lastIndex) {
                    val nx = x + directions[i-1]
                    val ny = y + directions[i]
                    if (nx in 0..maze[0].lastIndex &&
                            ny in 0..maze.lastIndex &&
                            maze[ny][nx] == '.') {
                        if (nx == 0 || 
                                ny == 0 || 
                                nx == maze[0].lastIndex || 
                                ny == maze.lastIndex) return steps
                        maze[ny][nx] = 'x'
                        queue.add(nx to ny)
                    }
                }
            }
            steps++
        }
        
        return -1
    }
```
Just do BFS.
* we can modify input matrix, so we can use it as visited array

Complexity: O(N), N - number of cells in maze
Memory: O(N)

# 20.11.2022
[https://leetcode.com/problems/basic-calculator/](https://leetcode.com/problems/basic-calculator/) hard

```kotlin
    fun calculate(s: String): Int {
        var i = 0
        var sign = 1
        var eval = 0
        while (i <= s.lastIndex) {
            val chr = s[i]
            if (chr == '(') {
                //find the end
                var countOpen = 0
                for (j in i..s.lastIndex) {
                    if (s[j] == '(') countOpen++
                    if (s[j] == ')') countOpen--
                    if (countOpen == 0) {
                        //evaluate substring
                        eval += sign * calculate(s.substring(i+1, j)) // [a b)
                        sign = 1
                        i = j
                        break
                    }
                }
            } else if (chr == '+') {
                sign = 1
            } else if (chr == '-') {
                sign = -1
            } else if (chr == ' ') {
                //nothing
            } else {
                var num = (s[i] - '0').toInt()
                for (j in (i+1)..s.lastIndex) {
                    if (s[j].isDigit()) {
                        num = num * 10 + (s[j] - '0').toInt()
                        i = j
                    } else  break
                }
                eval += sign * num
                sign = 1
            }
            i++
        }
        return eval
    }
```
This is a classic calculator problem, nothing special.
* be careful with the indexes

Complexity: O(N)
Memory: O(N), because of the recursion, worst case is all the input is brackets

# 19.11.2022
[https://leetcode.com/problems/erect-the-fence/](https://leetcode.com/problems/erect-the-fence/) hard

```kotlin
    fun outerTrees(trees: Array<IntArray>): Array<IntArray> {
        if (trees.size <= 3) return trees
        trees.sortWith(Comparator { a, b -> if (a[0]==b[0]) a[1]-b[1] else a[0] - b[0]} )
        fun cmp(a: IntArray, b: IntArray, c: IntArray): Int {
            val xab = b[0] - a[0]
            val yab = b[1] - a[1]
            val xbc = c[0] - b[0]
            val ybc = c[1] - b[1]
            return xab*ybc - yab*xbc
        }
        val up = mutableListOf<IntArray>()
        val lo = mutableListOf<IntArray>()
        trees.forEach { curr ->
            while(up.size >= 2 && cmp(up[up.size-2], up[up.size-1], curr) < 0) up.removeAt(up.lastIndex)
            while(lo.size >= 2 && cmp(lo[lo.size-2], lo[lo.size-1], curr) > 0) lo.removeAt(lo.lastIndex)
            up.add(curr)
            lo.add(curr)
        }
        return (up+lo).distinct().toTypedArray()
    }
```
This is an implementation of the [Andrew's monotonic chain](https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain) algorithm.
* need to remember vector algebra equation for ccw (counter clockwise) check (see [here](https://en.wikipedia.org/wiki/Curve_orientation#Orientation_of_a_simple_polygon))
* don't forget to sort by x and then by y

Complexity: O(Nlog(N))
Memory: O(N)

# 18.11.2022
[https://leetcode.com/problems/ugly-number/](https://leetcode.com/problems/ugly-number/) easy

```kotlin
    fun isUgly(n: Int): Boolean {
        if (n <= 0) return false
        var x = n
        while(x%2==0) x = x/2
        while(x%3==0) x = x/3
        while(x%5==0) x = x/5
        return x == 1
    }
```
There is also a clever math solution, but I don't understand it yet.

Complexity: O(log(n))
Memory: O(1)

# 17.11.2022
[https://leetcode.com/problems/rectangle-area/](https://leetcode.com/problems/rectangle-area/) middle

```kotlin
class Solution {
    class P(val x: Int, val y: Int)
    class Rect(val l: Int, val t: Int, val r: Int, val b: Int) {
        val corners = arrayOf(P(l, t), P(l, b), P(r, t), P(r, b))
        val s = (r - l) * (t - b)
        fun contains(p: P) = p.x in l..r && p.y in b..t
        fun intersect(o: Rect): Rect {
            val allX = intArrayOf(l, r, o.l, o.r).apply { sort() }
            val allY = intArrayOf(b, t, o.b, o.t).apply { sort() }
            val r = Rect(allX[1], allY[2], allX[2], allY[1])
            return if (r.corners.all { contains(it) && o.contains(it)}) 
                r else Rect(0,0,0,0)
        }
    }
    
    fun computeArea(ax1: Int, ay1: Int, ax2: Int, ay2: Int, bx1: Int, by1: Int, bx2: Int, by2: Int): Int {
        val r1 = Rect(ax1, ay2, ax2, ay1)
        val r2 = Rect(bx1, by2, bx2, by1)
        return r1.s + r2.s -  r1.intersect(r2).s
    }
}
```
This is an OOP problem. One trick to write intersection function is to notice that all corners of intersection rectangle
must be inside both rectangles. Also, intersection rectangle formed from middle coordinates of all corners sorted by x and y.

Complexity: O(1)
Memory: O(1)

# 16.11.2022
[https://leetcode.com/problems/guess-number-higher-or-lower/](https://leetcode.com/problems/guess-number-higher-or-lower/) easy

```kotlin
    override fun guessNumber(n:Int):Int {
       var lo = 1
       var hi = n
       while(lo <= hi) {
           val pick = lo + (hi - lo)/2
           val answer = guess(pick)
           if (answer == 0) return pick
           if (answer == -1) hi = pick - 1
           else lo = pick + 1
       }
       return lo
    }
```
This is a classic binary search algorithm. 
The best way of writing it is:
* use safe mid calculation (lo + (hi - lo)/2)
* use lo <= hi instead of lo < hi and mid+1/mid-1 instead of mid
  
Complexity: O(log(N))
Memory: O(1)


# 15.11.2022
[https://leetcode.com/problems/count-complete-tree-nodes/](https://leetcode.com/problems/count-complete-tree-nodes/) medium
```
       x
     *   x
   *   *   x
 *   x   *   x
* x x x x * x x
          \
          on each node we can check it's left and right depths
          this only takes us O(logN) time on each step
          there are logN steps in total (height of the tree)
          so the total time complexity is O(log^2(N))
```

```kotlin
    fun countNodes(root: TreeNode?): Int {
        var hl = 0
        var node = root  
        while (node != null) {
            node = node.left
            hl++
        }
        var hr = 0
        node = root  
        while (node != null) {
            node = node.right
            hr++
        }
        return when {
            hl == 0 -> 0 
            hl == hr -> (1 shl hl) - 1
            else -> 1  + 
            (root!!.left?.let {countNodes(it)}?:0) +
            (root!!.right?.let {countNodes(it)}?:0)
        }
    }
```
Complexity: O(log^2(N))
Memory: O(logN)

# 14.11.2022
[https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/](https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/) medium

From observing the problem, we can see, that the task is in fact is to find an isolated islands:
```
        // * 3 *         * 3 *        * * *
        // 1 2 *    ->   * * *   or   1 * *
        // * * 4         * * 4        * * 4

        // * 3 *         * * *
        // 1 2 5    ->   * * *
        // * * 4         * * 4
```
```kotlin
    fun removeStones(stones: Array<IntArray>): Int {
        val uf = IntArray(stones.size) { it }
        var rootsCount = uf.size
        fun root(a: Int): Int {
            var x = a
            while (uf[x] != x) x = uf[x]
            return x
        }
        fun union(a: Int, b: Int) {
           val rootA = root(a) 
           val rootB = root(b)
           if (rootA != rootB) {
               uf[rootA] = rootB
               rootsCount--
           }
        }
        val byY = mutableMapOf<Int, MutableList<Int>>()
        val byX = mutableMapOf<Int, MutableList<Int>>()
        stones.forEachIndexed { i, st ->
            byY.getOrPut(st[0], { mutableListOf() }).add(i)
            byX.getOrPut(st[1], { mutableListOf() }).add(i)
        }
        byY.values.forEach { list ->
            if (list.size > 1) 
                for (i in 1..list.lastIndex) union(list[0], list[i])
        }
        byX.values.forEach { list ->
            if (list.size > 1) 
                for (i in 1..list.lastIndex) union(list[0], list[i])
        }
        return stones.size - rootsCount
    }
```
Complexity: O(N)
Memory: O(N)

# 13.11.2022
[https://leetcode.com/problems/reverse-words-in-a-string/](https://leetcode.com/problems/reverse-words-in-a-string/) medium

A simple trick: reverse all the string, then reverse each word.

```kotlin
    fun reverseWords(s: String): String {
        val res = StringBuilder()
        val curr = Stack<Char>()
        (s.lastIndex downTo 0).forEach { i ->
            val c = s[i]
            if (c in '0'..'z') curr.push(c)
            else if (curr.isNotEmpty()) {
                if (res.length > 0) res.append(' ')
                while (curr.isNotEmpty()) res.append(curr.pop())
            }
        }
        if (curr.isNotEmpty() && res.length > 0) res.append(' ')
        while (curr.isNotEmpty()) res.append(curr.pop())
        return res.toString()
    }
```
Complexity: O(N)
Memory: O(N) - there is no O(1) solution for string in JVM

# 12.11.2022
[https://leetcode.com/problems/find-median-from-data-stream/](https://leetcode.com/problems/find-median-from-data-stream/) hard

To find the median we can maintain two heaps: smaller and larger. One decreasing and one increasing.
Peeking the top from those heaps will give us the median.
```
    //  [5 2 0] [6 7 10]
    //  dec     inc
    //   ^ peek  ^ peek
```

```kotlin
class MedianFinder() {
    val queDec = PriorityQueue<Int>(reverseOrder())
    val queInc = PriorityQueue<Int>()
    fun addNum(num: Int) {
        if (queDec.size == queInc.size) {
            queInc.add(num)
            queDec.add(queInc.poll())
        } else {
            queDec.add(num)
            queInc.add(queDec.poll())
        }
    }

    fun findMedian(): Double = if (queInc.size == queDec.size)
            (queInc.peek() + queDec.peek()) / 2.0
        else 
            queDec.peek().toDouble()
}
```
Complexity: O(NlogN)
Memory: O(N)
# 11.11.2022
[https://leetcode.com/problems/remove-duplicates-from-sorted-array/](https://leetcode.com/problems/remove-duplicates-from-sorted-array/) easy

Just do what is asked. Keep track of the pointer to the end of the "good" part.
```
    fun removeDuplicates(nums: IntArray): Int {
        var k = 0
        for (i in 1..nums.lastIndex) {
            if (nums[k] != nums[i]) nums[++k] = nums[i]
        }
        
        return k + 1
    }
```
Complexity: O(N)
Memory: O(1)

# 10.11.2022
[https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/) easy

Solution:
```
    fun removeDuplicates(s: String): String {
        val stack = Stack<Char>()
        s.forEach { c ->
            if (stack.isNotEmpty() && stack.peek() == c) {
                stack.pop()
            } else {
                stack.push(c)
            }
        }
        return stack.joinToString("")
    }
```
Explanation: Just scan symbols one by one and remove duplicates from the end.
Complexity: O(N)
Memory: O(N)

# 9.11.2022
[https://leetcode.com/problems/online-stock-span/](https://leetcode.com/problems/online-stock-span/) medium

So, we need to keep increasing sequence of numbers, increasing/decreasing stack will help.
Consider example, this is how decreasing stack will work
```
        // 100   [100-1]                            1
        // 80    [100-1, 80-1]                      1
        // 60    [100-1, 80-1, 60-1]                1
        // 70    [100-1, 80-1, 70-2] + 60           2
        // 60    [100-1, 80-1, 70-2, 60-1]          1
        // 75    [100-1, 80-1, 75-4] + 70-2+60-1    4
        // 85    [100-1, 85-6] 80-1+75-4            6
```
Solution:
```
class StockSpanner() {
    val stack = Stack<Pair<Int,Int>>()

    fun next(price: Int): Int {
        // 100   [100-1]                            1
        // 80    [100-1, 80-1]                      1
        // 60    [100-1, 80-1, 60-1]                1
        // 70    [100-1, 80-1, 70-2] + 60           2
        // 60    [100-1, 80-1, 70-2, 60-1]          1
        // 75    [100-1, 80-1, 75-4] + 70-2+60-1    4
        // 85    [100-1, 85-6] 80-1+75-4            6
       var span = 1
       while(stack.isNotEmpty() && stack.peek().first <= price) {
          span += stack.pop().second 
       } 
       stack.push(price to span)
       return span
    }

}
```
Complexity: O(N)
Memory: O(N)

# 8.11.2022
[https://leetcode.com/problems/make-the-string-great/](https://leetcode.com/problems/make-the-string-great/) easy

```
    fun makeGood(s: String): String {
        var ss = s.toCharArray()
        var finished = false
        while(!finished) {
            finished = true
            for (i in 0 until s.lastIndex) {
                if (ss[i] == '.') continue
                var j = i+1
                while(j <= s.lastIndex && ss[j] == '.') {
                    j++
                    continue
                }
                if (j == s.length) break
                
                var a = ss[i]
                var b = ss[j]
                if (a != b && Character.toLowerCase(a) == 
                        Character.toLowerCase(b)) {
                    ss[i] = '.'
                    ss[j] = '.'
                    finished = false
                }
            }
        }
        return ss.filter { it != '.' }.joinToString("")
    }

```
Explanation:
The simplest solution is just to simulate all the process, as input string is just 100 symbols.

Speed: O(n^2)
Memory: O(n)

# 7.11.2022
[https://leetcode.com/problems/maximum-69-number/](https://leetcode.com/problems/maximum-69-number/) easy

```
    fun maximum69Number (num: Int): Int {
        var n = num
        if (6666 <= n && n <= 6999) return num + 3000
        if (n > 9000) n -= 9000
        if (666 <= n && n <= 699) return num + 300
        if (n > 900) n -= 900
        if (66 <= n && n <= 69) return num + 30
        if (n > 90) n -= 90
        if (6 == n) return num + 3
        return num
    }
```
Explanation:
The simplest implementations would be converting to array of digits, replacing the first and converting back. 
However we can observe that numbers are in range 6-9999, so we can hardcode some logic.

Speed: O(1), Memory: O(1)

# 6.11.2022
[https://leetcode.com/problems/orderly-queue/](https://leetcode.com/problems/orderly-queue/) hard

```

    fun orderlyQueue(s: String, k: Int): String {
        val chrs = s.toCharArray()
        if (k == 1) {
            var smallest = s
            for (i in 0..s.lastIndex) {
                val prefix = s.substring(0, i)
                val suffix = s.substring(i)
                val ss = suffix + prefix
                if (ss.compareTo(smallest) < 0) smallest = ss
            }
            return smallest
        } else {
            chrs.sort()
            return String(chrs)
        }
    }

O(n^2)
```
Explanation:
One idea that come to my mind is: if k >= 2 then you basically can swap any adjacent elements. That means you can actually sort all the characters.

Speed: O(n^2), Memory: O(n)
    
# 6.11.2022
[https://leetcode.com/problems/word-search-ii/](https://leetcode.com/problems/word-search-ii/) hard

Solution [kotlin]
```

    class Node {
        val next = Array<Node?>(26) { null }
        var word: String?  = null
        operator fun invoke(c: Char): Node {
            val ind = c.toInt() - 'a'.toInt()
            if (next[ind] == null) next[ind] = Node()
            return next[ind]!!
        } 
        operator fun get(c: Char) = next[c.toInt() - 'a'.toInt()]
    }
    fun findWords(board: Array<CharArray>, words: Array<String>): List<String> {
        val trie = Node()
        
        words.forEach { w ->
            var t = trie
            w.forEach { t = t(it) }
            t.word = w
        }
        
        val result = mutableSetOf<String>()
        fun dfs(y: Int, x: Int, t: Node?, visited: MutableSet<Int>) {
           if (t == null || y < 0 || x < 0 
               || y >= board.size || x >= board[0].size 
               || !visited.add(100 * y + x)) return
           t[board[y][x]]?.let {
               it.word?.let {  result.add(it)  }
                dfs(y-1, x, it, visited)
                dfs(y+1, x, it, visited)
                dfs(y, x-1, it, visited)
                dfs(y, x+1, it, visited)
           }
           visited.remove(100 * y + x)
        }
        board.forEachIndexed { y, row ->
            row.forEachIndexed { x, c ->
                dfs(y, x, trie, mutableSetOf<Int>())
            }
        }
        return result.toList()
    }

```
Explanation:
Use trie + dfs
1. Collect all the words into the Trie
2. Search deeply starting from all the cells and advancing trie nodes
3. Collect if node is the word
4. Use set to avoid duplicates

Speed: O(wN + M), w=10, N=10^4, M=12^2 , Memory O(26w + N) 

# 4.11.2022
[https://leetcode.com/problems/reverse-vowels-of-a-string/](https://leetcode.com/problems/reverse-vowels-of-a-string/) easy

Solution [kotlin]
```
    fun reverseVowels(s: String): String {
        val vowels = setOf('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
        var chrs = s.toCharArray()
        var l = 0
        var r = chrs.lastIndex
        while(l < r) {
            while(l<r && chrs[l] !in vowels) l++
            while(l<r && chrs[r] !in vowels) r--
            if (l < r) chrs[l] = chrs[r].also { chrs[r] = chrs[l] }
            r--
            l++
        }
        return String(chrs)
    }
```
Explanation:
Straightforward solution : use two pointers method and scan from the both sides.

Speed: O(N), Memory O(N)

# 3.11.2022
[https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/](https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/) medium

Solution [kotlin]
```
fun longestPalindrome(words: Array<String>): Int {
        var singles = 0
        var mirrored = 0
        var uneven = 0
        var unevenSum = 0
        val visited = mutableMapOf<String, Int>()
        words.forEach { w ->  visited[w] = 1 + visited.getOrDefault(w, 0) }
        visited.forEach { w, wCount ->
            if (w[0] == w[1]) {
                if (wCount %2 == 0) {
                    singles += wCount*2
                } else {
                    // a b -> a
                    // a b a -> aba 2a + 1b = 2 + 1
                    // a b a b -> abba 2a + 2b = 2+2
                    // a b a b a -> baaab 3a + 2b = 3+2
                    // a b a b a b -> baaab 3a + 3b = 3+2 (-1)
                    // a b a b a b a -> aabbbaa 4a+3b=4+3
                    // a b a b a b a b -> aabbbbaa 4a+4b=4+4
                    // 5a+4b = 2+5+2
                    // 5a+5b = 2+5+2 (-1)
                    // 1c + 2b + 2a = b a c a b
                    // 1c + 3b + 2a =
                    // 1c + 3b + 4a = 2a + 3b + 2a
                    // 5d + 3a + 3b + 3c = a b c 5d c b a = 11 
                    uneven++
                    unevenSum += wCount
                }
            } else {
                val matchingCount = visited[w.reversed()] ?:0
                mirrored += minOf(wCount, matchingCount)*2
            }
        }
        val unevenCount = if (uneven == 0) 0 else 2*(unevenSum - uneven + 1)
        return singles + mirrored + unevenCount
    }
```
Explanation:
This is a counting task, can be solved linearly.
There are 3 cases: 
1. First count mirrored elements, "ab" <-> "ba", they all can be included to the result
2. Second count doubled letters "aa", "bb". Notice, that if count is even, they also can be splitted by half and all included. 
3. The only edge case is uneven part. The law can be derived by looking at the examples

Speed: O(N), Memory O(N)

# 2.11.2022
[https://leetcode.com/problems/minimum-genetic-mutation/](https://leetcode.com/problems/minimum-genetic-mutation/) medium

Solution [kotlin]
```
    fun minMutation(start: String, end: String, bank: Array<String>): Int {
        val wToW = mutableMapOf<Int, MutableList<Int>>()
        fun searchInBank(i1: Int, w1: String) {
            bank.forEachIndexed { i2, w2 ->
                if (w1 != w2) {
                    var diffCount = 0
                    for (i in 0..7) {
                        if (w1[i] != w2[i]) diffCount++
                    }
                    if (diffCount == 1) {
                       wToW.getOrPut(i1, { mutableListOf() }).add(i2)
                       wToW.getOrPut(i2, { mutableListOf() }).add(i1)
                    }
                }
            }
        }
        bank.forEachIndexed { i1, w1 -> searchInBank(i1, w1) }
        searchInBank(-1, start)
        val queue = ArrayDeque<Int>()
        queue.add(-1)
        var steps = 0
        while(queue.isNotEmpty()) {
            repeat(queue.size) {
                val ind = queue.poll()
                val word = if (ind == -1) start else bank[ind]
                if (word == end) return steps
                wToW[ind]?.let { siblings ->
                    siblings.forEach { queue.add(it) }
                }
            }
            steps++
            if (steps > bank.size + 1) return -1
        }
        return -1
    }
```
Explanation:
1. make graph
2. BFS in it
3. stop search if count > bank, or we can use visited map

Speed: O(wN^2), Memory O(N)

# 1.11.2022
[https://leetcode.com/problems/where-will-the-ball-fall/](https://leetcode.com/problems/where-will-the-ball-fall/) medium

Solution [kotlin]
```
    fun findBall(grid: Array<IntArray>): IntArray {
        var indToBall = IntArray(grid[0].size) { it }
        var ballToInd = IntArray(grid[0].size) { it }
        grid.forEach { row ->
            var nextIndToBall = IntArray(grid[0].size) { -1 }
            var nextBallToInd = IntArray(grid[0].size) { -1 }
            for (i in 0..row.lastIndex) {
                val currBall = indToBall[i]
                if (currBall != -1) {
                    val isCorner = row[i] == 1 
                    &&  i<row.lastIndex 
                    && row[i+1] == -1
                    || row[i] == -1
                    && i > 0
                    && row[i-1] == 1
                    
                    val newInd = i + row[i]
                    if (!isCorner && newInd >= 0 && newInd <= row.lastIndex) {
                        nextIndToBall[newInd] = currBall
                        nextBallToInd[currBall] = newInd
                    } 
                }
            }
            indToBall = nextIndToBall
            ballToInd = nextBallToInd
        }
        return ballToInd
    }
```
Explanation:
This is a geometry problem, but seeing the pattern might help. We can spot that each row is an action sequence: -1 -1 -1 shifts balls left, and 1 1 1 shifts balls to the right. Corners can be formed only with -1 1 sequence.  

# 31.10.2022
[https://leetcode.com/problems/toeplitz-matrix/](https://leetcode.com/problems/toeplitz-matrix/) easy

Solution [kotlin]
```
    fun isToeplitzMatrix(matrix: Array<IntArray>): Boolean =
        matrix
        .asSequence()
        .windowed(2)
        .all { (prev, curr) -> prev.dropLast(1) == curr.drop(1) }
```
Explanation:
just compare adjacent rows, they must have an equal elements except first and last
