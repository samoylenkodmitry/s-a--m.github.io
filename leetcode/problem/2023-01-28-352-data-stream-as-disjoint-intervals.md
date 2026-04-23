---
layout: leetcode-entry
title: "352. Data Stream as Disjoint Intervals"
permalink: "/leetcode/problem/2023-01-28-352-data-stream-as-disjoint-intervals/"
leetcode_ui: true
entry_slug: "2023-01-28-352-data-stream-as-disjoint-intervals"
---

[352. Data Stream as Disjoint Intervals](https://leetcode.com/problems/data-stream-as-disjoint-intervals/description/) hard

[blog post](https://leetcode.com/problems/data-stream-as-disjoint-intervals/solutions/3108727/kotlin-linked-list/)

```kotlin
class SummaryRanges() {
    data class Node(var start: Int, var end: Int, var next: Node? = null)

    val root = Node(-1, -1)

    fun mergeWithNext(n: Node?): Boolean {
        if (n == null) return false
        val curr = n
        val next = n.next
        if (next == null) return false
        val nextNext = next.next
        if (next.start - curr.end <= 1) {
            curr.end = next.end
            curr.next = nextNext
            return true
        }
        return false
    }

    fun addNum(value: Int) {
        var n = root
        while (n.next != null && n.next!!.start < value) n = n.next!!
        if (value in n.start..n.end) return
        n.next = Node(value, value, n.next)
        if (n != root && mergeWithNext(n))
            mergeWithNext(n)
        else
            mergeWithNext(n.next)
    }

    fun getIntervals(): Array<IntArray> {
        val list = mutableListOf<IntArray>()
        var n = root.next
        while (n != null) {
            list.add(intArrayOf(n.start, n.end))
            n = n.next
        }
        return list.toTypedArray()
    }

}

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/100
#### Intuition
In Kotlin there is no way around to avoid the O(n) time of an operation while building the result array.
And there is no way to insert to the middle of the array in a less than O(n) time.
So, the only way is to use the linked list, and to walk it linearly.

#### Approach
* careful with merge
#### Complexity
- Time complexity:
  $$O(IN)$$, I - number of the intervals
- Space complexity:
  $$O(I)$$

