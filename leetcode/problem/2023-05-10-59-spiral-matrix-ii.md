---
layout: leetcode-entry
title: "59. Spiral Matrix II"
permalink: "/leetcode/problem/2023-05-10-59-spiral-matrix-ii/"
leetcode_ui: true
entry_slug: "2023-05-10-59-spiral-matrix-ii"
---

[59. Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/description/) medium

```kotlin

fun generateMatrix(n: Int): Array<IntArray> = Array(n) { IntArray(n) }.apply {
    var dir = 0
    var dxdy = arrayOf(0, 1, 0, -1)
    var x = 0
    var y = 0
    val nextX = { x + dxdy[(dir + 1) % 4] }
    val nextY = { y + dxdy[dir] }
    val valid = { x: Int, y: Int -> x in 0..n-1 && y in 0..n-1 && this[y][x] == 0 }

    repeat (n * n) {
        this[y][x] = it + 1
        if (!valid(nextX(), nextY())) dir = (dir + 1) % 4
        x = nextX()
        y = nextY()
    }
}

```

[blog post](https://leetcode.com/problems/spiral-matrix-ii/solutions/3506921/kotlin-a-robot/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-10052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/208
#### Intuition
Just implement what is asked. Let's have the strategy of a robot: move it in one direction until it hits a wall, then change the direction.

#### Approach
* to detect an empty cell, we can check it for `== 0`
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

