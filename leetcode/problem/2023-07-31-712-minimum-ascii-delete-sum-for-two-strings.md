---
layout: leetcode-entry
title: "712. Minimum ASCII Delete Sum for Two Strings"
permalink: "/leetcode/problem/2023-07-31-712-minimum-ascii-delete-sum-for-two-strings/"
leetcode_ui: true
entry_slug: "2023-07-31-712-minimum-ascii-delete-sum-for-two-strings"
---

[712. Minimum ASCII Delete Sum for Two Strings](https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/description/) medium
[blog post](https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/solutions/3840916/kotlin-dfs-memo/)
[substack](https://dmitriisamoilenko.substack.com/p/31072023-712-minimum-ascii-delete?sd=pf)
![image.png](/assets/leetcode_daily_images/d06babde.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/292

#### Problem TLDR

Minimum removed chars sum to make strings equal

#### Intuition

This is a known Dynamic Programming problem about the minimum edit distance. We can walk both strings and at each time choose what char to take and what to skip. The result is dependent only from the arguments, so can be cached.

#### Approach

Let's use DFS and memo.

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun minimumDeleteSum(s1: String, s2: String): Int {
      val cache = mutableMapOf<Pair<Int, Int>, Int>()
      fun dfs(p1: Int, p2: Int): Int = cache.getOrPut(p1 to p2) { when {
        p1 == s1.length && p2 == s2.length -> 0
        p1 == s1.length -> s2.drop(p2).map { it.toInt() }.sum()!!
        p2 == s2.length -> s1.drop(p1).map { it.toInt() }.sum()!!
        s1[p1] == s2[p2] -> dfs(p1 + 1, p2 + 1)
        else -> minOf(s1[p1].toInt() + dfs(p1 + 1, p2), s2[p2].toInt() + dfs(p1, p2 + 1))
      } }
      return dfs(0, 0)
    }

```

