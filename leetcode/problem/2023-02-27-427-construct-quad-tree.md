---
layout: leetcode-entry
title: "427. Construct Quad Tree"
permalink: "/leetcode/problem/2023-02-27-427-construct-quad-tree/"
leetcode_ui: true
entry_slug: "2023-02-27-427-construct-quad-tree"
---

[427. Construct Quad Tree](https://leetcode.com/problems/construct-quad-tree/description/) medium

[blog post](https://leetcode.com/problems/construct-quad-tree/solutions/3235370/kotlin-dfs/)

```kotlin

fun construct(grid: Array<IntArray>): Node? {
    if (grid.isEmpty()) return null
    fun dfs(xMin: Int, xMax: Int, yMin: Int, yMax: Int): Node? {
        if (xMin == xMax) return Node(grid[yMin][xMin] == 1, true)
        val xMid = xMin + (xMax - xMin) / 2
        val yMid = yMin + (yMax - yMin) / 2
        return Node(false, false).apply {
            topLeft = dfs(xMin, xMid, yMin, yMid)
            topRight = dfs(xMid + 1, xMax, yMin, yMid)
            bottomLeft = dfs(xMin, xMid, yMid + 1, yMax)
            bottomRight = dfs(xMid + 1, xMax, yMid + 1, yMax)
            if (topLeft!!.isLeaf && topRight!!.isLeaf
            && bottomLeft!!.isLeaf && bottomRight!!.isLeaf) {
                if (topLeft!!.`val` == topRight!!.`val`
                && topRight!!.`val` == bottomLeft!!.`val`
                && bottomLeft!!.`val` == bottomRight!!.`val`) {
                    `val` = topLeft!!.`val`
                    isLeaf = true
                    topLeft = null
                    topRight = null
                    bottomLeft = null
                    bottomRight = null
                }
            }
        }
    }
    return dfs(0, grid[0].lastIndex, 0, grid.lastIndex)
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/131
#### Intuition
We can construct the tree using DFS and divide and conquer technique. Build four nodes, then check if all of them are equal leafs.

#### Approach
* use inclusive ranges to simplify the code
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

