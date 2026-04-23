---
layout: leetcode-entry
title: "228. Summary Ranges"
permalink: "/leetcode/problem/2023-06-12-228-summary-ranges/"
leetcode_ui: true
entry_slug: "2023-06-12-228-summary-ranges"
---

![image.png](/assets/leetcode_daily_images/0edb7543.webp)
[228. Summary Ranges](https://leetcode.com/problems/summary-ranges/description/) easy
[blog post](https://leetcode.com/problems/summary-ranges/solutions/3627478/kotlin-fold/)
[substack](https://dmitriisamoilenko.substack.com/p/12062023-228-summary-ranges?sd=pf)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/243
#### Problem TLDR
Fold continues ranges in a sorted array `1 2 3 5` -> `1->3, 5`
#### Intuition
Scan from start to end, modify the last interval or add a new one.

#### Approach
Let's write a Kotlin one-liner

#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun summaryRanges(nums: IntArray): List<String> = nums
    .fold(mutableListOf<IntArray>()) { r, t ->
        if (r.isEmpty() || r.last()[1] + 1 < t) r += intArrayOf(t, t)
        else r.last()[1] = t
        r
    }
    .map { (f, t) -> if (f == t) "$f" else "$f->$t"}

```

