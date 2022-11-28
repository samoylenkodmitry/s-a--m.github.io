---
layout: post
title: Daily leetcode challenge
---

# Daily leetcode challenge
You can join me and discuss in the Telegram channel [https://t.me/leetcode_daily_unstoppable](https://t.me/leetcode_daily_unstoppable)
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
