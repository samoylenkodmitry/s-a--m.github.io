---
layout: leetcode-entry
title: "1420. Build Array Where You Can Find The Maximum Exactly K Comparisons"
permalink: "/leetcode/problem/2023-10-07-1420-build-array-where-you-can-find-the-maximum-exactly-k-comparisons/"
leetcode_ui: true
entry_slug: "2023-10-07-1420-build-array-where-you-can-find-the-maximum-exactly-k-comparisons"
---

[1420. Build Array Where You Can Find The Maximum Exactly K Comparisons](https://leetcode.com/problems/build-array-where-you-can-find-the-maximum-exactly-k-comparisons/description/) hard
[blog post](https://leetcode.com/problems/build-array-where-you-can-find-the-maximum-exactly-k-comparisons/solutions/4140362/kotlin-dfs-cache/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/7102023-1420-build-array-where-you?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/a9c26816.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/362

#### Problem TLDR

Count possible arrays of n `1..m` values increasing `k` times

#### Intuition

First, try to write down some examples of arrays. There are some laws of how the number of arrays grows.

Next, use hint :)

Then just write Depth First Search of all possible numbers for each position and count how many times numbers grows. Stop search when it is bigger than `k` times. The result can be cached.

#### Approach

* use Long to avoid overflows

#### Complexity

- Time complexity:
$$O(nkm^2)$$, nkm - is a search depth, and another m for internal loop

- Space complexity:
$$O(nkm)$$

#### Code

```kotlin

    fun numOfArrays(n: Int, m: Int, k: Int): Int {
      val mod = 1_000_000_007L
      val dp = Array(n) { Array(m + 1) { Array(k + 1) { -1L } } }
      fun dfs(i: Int, max: Int, c: Int): Long =
        if (c > k) 0L
        else if (i == n) { if (c == k) 1L else 0L }
        else dp[i][max][c].takeIf { it >= 0 } ?: {
          var sum = (max * dfs(i + 1, max, c)) % mod
          for (x in (max + 1)..m)
            sum = (sum + dfs(i + 1, x, c + 1)) % mod
          sum
        }().also { dp[i][max][c] = it }
      return dfs(0, 0, 0).toInt()
    }

```

