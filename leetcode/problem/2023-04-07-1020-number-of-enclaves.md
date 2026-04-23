---
layout: leetcode-entry
title: "1020. Number of Enclaves"
permalink: "/leetcode/problem/2023-04-07-1020-number-of-enclaves/"
leetcode_ui: true
entry_slug: "2023-04-07-1020-number-of-enclaves"
---

[1020. Number of Enclaves](https://leetcode.com/problems/number-of-enclaves/description/) medium

[blog post](https://leetcode.com/problems/number-of-enclaves/solutions/3388636/kotlin-dfs/)

```kotlin

fun numEnclaves(grid: Array<IntArray>): Int {
    val visited = HashSet<Pair<Int, Int>>()
    fun dfs(x: Int, y: Int): Int {
        return if (x < 0 || y < 0 || x > grid[0].lastIndex || y > grid.lastIndex) 0
        else if (grid[y][x] == 1 && visited.add(x to y))
        1 + dfs(x - 1, y) + dfs(x + 1, y) + dfs(x, y - 1) + dfs(x, y + 1)
        else 0
    }
    for (y in 0..grid.lastIndex) {
        dfs(0, y)
        dfs(grid[0].lastIndex, y)
    }
    for (x in 0..grid[0].lastIndex) {
        dfs(x, 0)
        dfs(x, grid.lastIndex)
    }
    var count = 0
    for (y in 0..grid.lastIndex)
    for(x in 0..grid[0].lastIndex)
    count += dfs(x, y)
    return count
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/173
#### Intuition
Walk count all the `1` cells using DFS and a visited set.
#### Approach
We can use `visited` set, or modify the grid or use Union-Find.
To exclude the borders, we can visit them first with DFS.
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

