---
layout: leetcode-entry
title: "1970. Last Day Where You Can Still Cross"
permalink: "/leetcode/problem/2023-06-30-1970-last-day-where-you-can-still-cross/"
leetcode_ui: true
entry_slug: "2023-06-30-1970-last-day-where-you-can-still-cross"
---

[1970. Last Day Where You Can Still Cross](https://leetcode.com/problems/last-day-where-you-can-still-cross/description/) hard
[blog post](https://leetcode.com/problems/last-day-where-you-can-still-cross/solutions/3698920/kotlin-union-find/)
[substack](https://dmitriisamoilenko.substack.com/p/30062023-1970-last-day-where-you?sd=pf)
![image.png](/assets/leetcode_daily_images/77cb954b.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/261
#### Problem TLDR
Last `day` matrix connected top-bottom when flooded each day at `cells[day]`
#### Intuition
One possible solution is to do a Binary Search in a days space, however it gives TLE.
Let's invert the problem: find the first day from the end where there is a connection top-bottom.
![image.png](/assets/leetcode_daily_images/c29b45f6.webp)
Now, `cells[day]` is a new ground. We can use Union-Find to connect ground cells.

#### Approach
* use sentinel cells for `top` and `bottom`
* use path compressing `uf[n] = x`

#### Complexity

- Time complexity:
$$O(an)$$, where `a` is a reverse Ackerman function

- Space complexity:
$$O(n)$$

#### Code

```kotlin

val uf = HashMap<Int, Int>()
fun root(x: Int): Int = if (uf[x] == null || uf[x] == x) x else root(uf[x]!!)
.also { uf[x] = it }
fun latestDayToCross(row: Int, col: Int, cells: Array<IntArray>) =
    cells.size - 1 - cells.reversed().indexOfFirst { (y, x) ->
        uf[y * col + x] = root(if (y == 1) 0 else if (y == row) 1 else y * col + x)
        sequenceOf(y to x - 1, y to x + 1, y - 1 to x, y + 1 to x)
        .filter { (y, x) -> y in 1..row && x in 1..col }
        .map { (y, x) -> y * col + x }
        .forEach { if (uf[it] != null) uf[root(y * col + x)] = root(it) }
        root(0) == root(1)
    }

```

