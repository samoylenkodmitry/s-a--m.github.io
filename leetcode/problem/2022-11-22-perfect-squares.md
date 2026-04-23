---
layout: leetcode-entry
title: "Perfect Squares"
permalink: "/leetcode/problem/2022-11-22-perfect-squares/"
leetcode_ui: true
entry_slug: "2022-11-22-perfect-squares"
---

[https://leetcode.com/problems/perfect-squares/](https://leetcode.com/problems/perfect-squares/) medium

```kotlin

    val cache = mutableMapOf<Int, Int>()
    fun numSquares(n: Int): Int {
        if (n < 0) return -1
        if (n == 0) return 0
        if (cache[n] != null) return cache[n]!!
        var min = Int.MAX_VALUE
        for (x in Math.sqrt(n.toDouble()).toInt() downTo 1) {
            val res = numSquares(n - x*x)
            if (res != -1) {
                min = minOf(min, 1 + res)
            }
        }
        if (min == Int.MAX_VALUE) min = -1
        cache[n] = min
        return min
    }

```

The problem gives stable answers for any argument n.
So, we can use memoization technique and search from the biggest square to the smallest one.

Complexity: O(Nsqrt(N))
Memory: O(N)

