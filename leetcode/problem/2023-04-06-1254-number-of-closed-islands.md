---
layout: leetcode-entry
title: "1254. Number of Closed Islands"
permalink: "/leetcode/problem/2023-04-06-1254-number-of-closed-islands/"
leetcode_ui: true
entry_slug: "2023-04-06-1254-number-of-closed-islands"
---

[1254. Number of Closed Islands](https://leetcode.com/problems/number-of-closed-islands/description/) medium

[blog post](https://leetcode.com/problems/number-of-closed-islands/solutions/3385170/kotlin-dfs/)

```kotlin

fun closedIsland(grid: Array<IntArray>): Int {
    val visited = HashSet<Pair<Int, Int>>()
    val seen = HashSet<Pair<Int, Int>>()

    fun dfs(x: Int, y: Int): Boolean {
        seen.add(x to y)
        if (x >= 0 && y >= 0 && x < grid[0].size && y < grid.size
        && grid[y][x] == 0 &&  visited.add(x to y)) {
            var isBorder = x == 0 || y == 0 || x == grid[0].lastIndex || y == grid.lastIndex
            isBorder = dfs(x - 1, y) || isBorder
            isBorder = dfs(x, y - 1) || isBorder
            isBorder = dfs(x + 1, y) || isBorder
            isBorder = dfs(x, y + 1) || isBorder
            return isBorder
        }
        return false
    }

    var count = 0
    for (y in 0..grid.lastIndex)
    for (x in 0..grid[0].lastIndex)
    if (grid[y][x] == 0 && seen.add(x to y) && !dfs(x, y)) count++
    return count
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/172
#### Intuition
Use hint #1, if we don't count islands on the borders, we get the result. Now, just count all connected `0` cells that didn't connect to the borders. We can use DFS or Union-Find.
#### Approach
DFS will solve the problem.
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

