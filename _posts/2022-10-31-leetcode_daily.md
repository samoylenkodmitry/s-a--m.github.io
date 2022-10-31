---
layout: post
title: Daily leetcode challenge
---

# Daily leetcode challenge
You can join and discuss in Telegram channel [t.me/leetcode_daily_unstoppable](t.me/leetcode_daily_unstoppable)

# Today
[https://leetcode.com/problems/toeplitz-matrix/](https://leetcode.com/problems/toeplitz-matrix/) easy

Solution [kotlin]
```
    fun isToeplitzMatrix(matrix: Array<IntArray>): Boolean =
        matrix
        .asSequence()
        .windowed(2)
        .all { (prev, curr) -> prev.dropLast(1) == curr.drop(1) }
```
Explanation:
just compare adjacent rows, they must have an equal elements except first and last
