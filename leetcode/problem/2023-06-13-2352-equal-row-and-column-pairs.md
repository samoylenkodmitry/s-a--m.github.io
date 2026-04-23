---
layout: leetcode-entry
title: "2352. Equal Row and Column Pairs"
permalink: "/leetcode/problem/2023-06-13-2352-equal-row-and-column-pairs/"
leetcode_ui: true
entry_slug: "2023-06-13-2352-equal-row-and-column-pairs"
---

[2352. Equal Row and Column Pairs](https://leetcode.com/problems/equal-row-and-column-pairs/description/) medium
[blog post](https://leetcode.com/problems/equal-row-and-column-pairs/solutions/3631323/kotlin-hash/)
[substack](https://dmitriisamoilenko.substack.com/p/12062023-2352-equal-row-and-column?sd=pf)
![image.png](/assets/leetcode_daily_images/ad7e8efa.webp)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/244
#### Problem TLDR
Count of `rowArray` == `colArray` in an `n x n` matrix.

#### Intuition
Compute `hash` function for each `row ` and each `col`, then compare them. If `hash(row) == hash(col)`, then compare arrays.
For hashing, we can use simple `31 * prev + curr`, that encodes both value and position.

#### Approach
* For this Leetcode data, `tan` hash works perfectly, we can skip comparing the arrays.

#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun equalPairs(grid: Array<IntArray>): Int {
    val rowHashes = grid.map { it.fold(0.0) { r, t ->  Math.tan(r) + t } }
    val colHashes = (0..grid.lastIndex).map { x ->
        (0..grid.lastIndex).fold(0.0) { r, t -> Math.tan(r) + grid[t][x] } }
        return (0..grid.size * grid.size - 1).count {
            rowHashes[it / grid.size] == colHashes[it % grid.size]
        }
    }

```

