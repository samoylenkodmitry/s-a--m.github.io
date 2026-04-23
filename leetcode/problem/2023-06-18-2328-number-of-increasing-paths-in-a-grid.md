---
layout: leetcode-entry
title: "2328. Number of Increasing Paths in a Grid"
permalink: "/leetcode/problem/2023-06-18-2328-number-of-increasing-paths-in-a-grid/"
leetcode_ui: true
entry_slug: "2023-06-18-2328-number-of-increasing-paths-in-a-grid"
---

[2328. Number of Increasing Paths in a Grid](https://leetcode.com/problems/number-of-increasing-paths-in-a-grid/description/) hard
[blog post](https://leetcode.com/problems/number-of-increasing-paths-in-a-grid/solutions/3651039/kotlin-dfs-memo/)
[substack](https://dmitriisamoilenko.substack.com/p/18062023-2328-number-of-increasing?sd=pf)
![image.png](/assets/leetcode_daily_images/48940540.webp)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/249
#### Problem TLDR
Count increasing paths in a matrix
#### Intuition
For every cell in a matrix, we can calculate how many increasing paths are starting from it. This result can be memorized. If we know the sibling's result, then we add it to the current if `curr > sibl`.

#### Approach
* use Depth-First search for the paths finding
* use `LongArray` for the memo
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun countPaths(grid: Array<IntArray>): Int {
    val m = 1_000_000_007L
    val counts = Array(grid.size) { LongArray(grid[0].size) }
    fun dfs(y: Int, x: Int): Long {
        return counts[y][x].takeIf { it != 0L } ?: {
            val v = grid[y][x]
            var sum = 1L
            if (x > 0 && v > grid[y][x - 1]) sum = (sum + dfs(y, x - 1)) % m
            if (y > 0 && v > grid[y - 1][x]) sum = (sum + dfs(y - 1, x)) % m
            if (y < grid.size - 1 && v > grid[y + 1][x]) sum = (sum + dfs(y + 1, x)) % m
            if (x < grid[0].size - 1 && v > grid[y][x + 1]) sum = (sum + dfs(y, x + 1)) % m
            sum
        }().also { counts[y][x] = it }
    }
    return (0 until grid.size * grid[0].size)
    .fold(0L) { r, t -> (r + dfs(t / grid[0].size, t % grid[0].size)) % m }
    .toInt()
}

```

