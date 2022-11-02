---
layout: post
title: Daily leetcode challenge
---

# Daily leetcode challenge
You can join me and discuss in the Telegram channel [t.me/leetcode_daily_unstoppable](t.me/leetcode_daily_unstoppable)

# Today
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
