---
layout: leetcode-entry
title: "2482. Difference Between Ones and Zeros in Row and Column"
permalink: "/leetcode/problem/2023-12-14-2482-difference-between-ones-and-zeros-in-row-and-column/"
leetcode_ui: true
entry_slug: "2023-12-14-2482-difference-between-ones-and-zeros-in-row-and-column"
---

[2482. Difference Between Ones and Zeros in Row and Column](https://leetcode.com/problems/difference-between-ones-and-zeros-in-row-and-column/description/) easy
[blog post](https://leetcode.com/problems/difference-between-ones-and-zeros-in-row-and-column/solutions/4402623/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14122023-2482-difference-between?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
![image.png](/assets/leetcode_daily_images/67147a68.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/438

#### Problem TLDR

diff[i][j] = onesRowi + onesColj - zerosRowi - zerosColj

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun onesMinusZeros(grid: Array<IntArray>): Array<IntArray> {
      val onesRow = grid.map { it.count { it == 1 } }
      val zerosRow = grid.map { it.count { it == 0 } }
      val onesCol = grid[0].indices.map { x -> grid.indices.count { grid[it][x] == 1 } }
      val zerosCol = grid[0].indices.map { x -> grid.indices.count { grid[it][x] == 0 } }
      return Array(grid.size) { y -> IntArray(grid[0].size) { x ->
        onesRow[y] + onesCol[x] - zerosRow[y] - zerosCol[x]
      }}
    }

```

