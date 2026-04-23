---
layout: leetcode-entry
title: "Where Will The Ball Fall"
permalink: "/leetcode/problem/2022-11-01-where-will-the-ball-fall/"
leetcode_ui: true
entry_slug: "2022-11-01-where-will-the-ball-fall"
---

[https://leetcode.com/problems/where-will-the-ball-fall/](https://leetcode.com/problems/where-will-the-ball-fall/) medium

Solution [kotlin]

```kotlin

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

