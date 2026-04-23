---
layout: leetcode-entry
title: "1046. Last Stone Weight"
permalink: "/leetcode/problem/2023-04-24-1046-last-stone-weight/"
leetcode_ui: true
entry_slug: "2023-04-24-1046-last-stone-weight"
---

[1046. Last Stone Weight](https://leetcode.com/problems/last-stone-weight/description/) easy

```kotlin

fun lastStoneWeight(stones: IntArray): Int =
with(PriorityQueue<Int>(compareByDescending { it } )) {
    stones.forEach { add(it) }
    while (size > 1) add(poll() - poll())
    if (isEmpty()) 0 else peek()
}

```

[blog post](https://leetcode.com/problems/last-stone-weight/solutions/3449145/kotlin-priority-queue/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-24042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/190
#### Intuition
Just run the simulation.

#### Approach
* use `PriorityQueue` with `compareByDescending`
#### Complexity
- Time complexity:
$$O(nlog(n))$$
- Space complexity:
$$O(n)$$

