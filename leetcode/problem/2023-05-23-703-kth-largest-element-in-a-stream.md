---
layout: leetcode-entry
title: "703. Kth Largest Element in a Stream"
permalink: "/leetcode/problem/2023-05-23-703-kth-largest-element-in-a-stream/"
leetcode_ui: true
entry_slug: "2023-05-23-703-kth-largest-element-in-a-stream"
---

[703. Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/description/) medium
[blog post](https://leetcode.com/problems/kth-largest-element-in-a-stream/solutions/3554138/kotlin-priority-queue/)
[substack](https://dmitriisamoilenko.substack.com/p/23052023-703-kth-largest-element?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/221
#### Problem TLDR
Kth largest
#### Intuition
We need to keep all values smaller than current largest kth element and can safely drop all other elements.
#### Approach
Use `PriorityQueue`.
#### Complexity
- Time complexity:
$$O(nlogk)$$
- Space complexity:
$$O(k)$$

#### Code

```kotlin

class KthLargest(val k: Int, nums: IntArray) {
    val pq = PriorityQueue<Int>(nums.toList())

        fun add(v: Int): Int = with (pq) {
            add(v)
            while (size > k) poll()
            peek()
        }
    }

```

