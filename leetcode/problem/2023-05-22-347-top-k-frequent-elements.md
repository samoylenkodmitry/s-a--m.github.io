---
layout: leetcode-entry
title: "347. Top K Frequent Elements"
permalink: "/leetcode/problem/2023-05-22-347-top-k-frequent-elements/"
leetcode_ui: true
entry_slug: "2023-05-22-347-top-k-frequent-elements"
---

[347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/) medium
[blog post](https://leetcode.com/problems/top-k-frequent-elements/solutions/3550637/kotlin-bucket-sort/)
[substack](https://dmitriisamoilenko.substack.com/p/22052023-347-top-k-frequent-elements?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/220
#### Problem TLDR
First `k` unique elements sorted by frequency.
#### Intuition
Group by frequency `1 1 1 5 5 -> 1:3, 5:2`, then bucket sort frequencies `2:5, 3:1`, then flatten and take first `k`.
#### Approach
* We can use [Kotlin collections api](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/-map/)
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun topKFrequent(nums: IntArray, k: Int): IntArray {
    val freq = nums.groupBy { it }.mapValues { it.value.size }
    val freqToNum = Array<MutableList<Int>>(nums.size + 1) { mutableListOf() }
    freq.forEach { (num, fr) -> freqToNum[nums.size + 1 - fr].add(num) }
    return freqToNum
        .filter { it.isNotEmpty() }
        .flatten()
        .take(k)
        .toIntArray()
}

```

