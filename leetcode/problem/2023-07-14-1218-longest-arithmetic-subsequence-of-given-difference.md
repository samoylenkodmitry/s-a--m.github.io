---
layout: leetcode-entry
title: "1218. Longest Arithmetic Subsequence of Given Difference"
permalink: "/leetcode/problem/2023-07-14-1218-longest-arithmetic-subsequence-of-given-difference/"
leetcode_ui: true
entry_slug: "2023-07-14-1218-longest-arithmetic-subsequence-of-given-difference"
---

[1218. Longest Arithmetic Subsequence of Given Difference](https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/description/) medium
[blog post](https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/solutions/3761793/kotlin-map/)
[substack](https://dmitriisamoilenko.substack.com/p/14072023-1218-longest-arithmetic?sd=pf)
![image.png](/assets/leetcode_daily_images/11f80563.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/275
#### Problem TLDR
Longest arithmetic `difference` subsequence
#### Intuition
Store the `next` value and the `length` for it.

#### Approach
We can use a `HashMap`
#### Complexity
- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun longestSubsequence(arr: IntArray, difference: Int): Int =
with(mutableMapOf<Int, Int>()) {
    arr.asSequence().map { x ->
        (1 + (this[x] ?: 0)).also { this[x + difference] = it }
    }.max()!!
}

```

