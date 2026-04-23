---
layout: leetcode-entry
title: "1337. The K Weakest Rows in a Matrix"
permalink: "/leetcode/problem/2023-09-18-1337-the-k-weakest-rows-in-a-matrix/"
leetcode_ui: true
entry_slug: "2023-09-18-1337-the-k-weakest-rows-in-a-matrix"
---

[1337. The K Weakest Rows in a Matrix](https://leetcode.com/problems/the-k-weakest-rows-in-a-matrix/description/) easy
[blog post](https://leetcode.com/problems/the-k-weakest-rows-in-a-matrix/solutions/4058213/kotlin-use-api/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18092023-1337-the-k-weakest-rows?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/e14ac843.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/343

#### Problem TLDR

`k` indices with smallest row sum in a binary matrix

#### Intuition

We can precompute row sums, then use a Priority Queue to find `k` smallest. However, just sorting all will also work.

#### Approach

Let's use Kotlin's collections API
* map
* filter
* sortedBy [https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/sorted-by.html](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/sorted-by.html)
* take
* toIntArray

#### Complexity
- Time complexity:
$$O(n^2logn)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun kWeakestRows(mat: Array<IntArray>, k: Int) = mat
        .map { it.filter { it == 1 }.sum() ?: 0 }
        .withIndex()
        .sortedBy { it.value }
        .map { it.index }
        .take(k)
        .toIntArray()

```

