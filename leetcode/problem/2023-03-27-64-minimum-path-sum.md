---
layout: leetcode-entry
title: "64. Minimum Path Sum"
permalink: "/leetcode/problem/2023-03-27-64-minimum-path-sum/"
leetcode_ui: true
entry_slug: "2023-03-27-64-minimum-path-sum"
---

[64. Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/description/) medium

[blog post](https://leetcode.com/problems/minimum-path-sum/solutions/3346543/kotlin-dfs-memo/)

```kotlin

    fun minPathSum(grid: Array<IntArray>): Int {
        val cache = mutableMapOf<Pair<Int, Int>, Int>()
        fun dfs(xy: Pair<Int, Int>): Int {
        return cache.getOrPut(xy) {
            val (x, y) = xy
            val curr = grid[y][x]
            if (x == grid[0].lastIndex && y == grid.lastIndex) curr else
            minOf(
            if (x < grid[0].lastIndex) curr + dfs((x + 1) to y)
            else Int.MAX_VALUE,
            if (y < grid.lastIndex) curr + dfs(x to (y + 1))
            else Int.MAX_VALUE
            )
        }
    }
    return dfs(0 to 0)
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/161
##### Intuition
On each cell of the grid, there is only one minimum path sum. So, we can memorize it. Or we can use a bottom up DP approach.

#### Approach
Use DFS + memo, careful with the ending condition.

#### Complexity
- Time complexity:
$$O(n^2)$$, where $$n$$ - matrix size
- Space complexity:
$$O(n^2)$$

