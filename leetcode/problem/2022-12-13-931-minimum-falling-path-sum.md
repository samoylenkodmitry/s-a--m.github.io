---
layout: leetcode-entry
title: "931. Minimum Falling Path Sum"
permalink: "/leetcode/problem/2022-12-13-931-minimum-falling-path-sum/"
leetcode_ui: true
entry_slug: "2022-12-13-931-minimum-falling-path-sum"
---

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

