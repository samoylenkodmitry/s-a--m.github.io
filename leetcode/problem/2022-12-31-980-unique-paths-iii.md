---
layout: leetcode-entry
title: "980. Unique Paths III"
permalink: "/leetcode/problem/2022-12-31-980-unique-paths-iii/"
leetcode_ui: true
entry_slug: "2022-12-31-980-unique-paths-iii"
---

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

