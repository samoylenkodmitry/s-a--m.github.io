---
layout: leetcode-entry
title: "54. Spiral Matrix"
permalink: "/leetcode/problem/2023-05-09-54-spiral-matrix/"
leetcode_ui: true
entry_slug: "2023-05-09-54-spiral-matrix"
---

[54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix/description/) medium

```kotlin

fun spiralOrder(matrix: Array<IntArray>): List<Int> = mutableListOf<Int>().apply {
    var x = 0
    var y = 0
    val dxy = arrayOf(0, 1, 0, -1)
    val borders = arrayOf(matrix[0].lastIndex, matrix.lastIndex, 0, 0)
    var dir = 0
    val moveBorder = { border: Int -> borders[border] += if (border < 2) -1 else 1 }
    repeat (matrix.size * matrix[0].size) {
        if ((if (dir % 2 == 0) x else y) == borders[dir]) {
            moveBorder((dir + 3) % 4)
            dir = (dir + 1) % 4
        }
        add(matrix[y][x])
        x += dxy[(dir + 1) % 4]
        y += dxy[dir]
    }
}

```

[blog post](https://leetcode.com/problems/spiral-matrix/solutions/3503485/kotlin-robot/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-9052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/207
#### Intuition
Just implement what is asked.
We can use a loop with four directions, or try to program `a robot` that will rotate after it hit a wall.

#### Approach
* do track the borders `left`, `top`, `right`, `bottom`
* use single direction variable `dir`
* move the wall after a robot walked parallel to it
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

