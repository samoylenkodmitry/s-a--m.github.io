---
layout: leetcode-entry
title: "Find Median From Data Stream"
permalink: "/leetcode/problem/2022-11-12-find-median-from-data-stream/"
leetcode_ui: true
entry_slug: "2022-11-12-find-median-from-data-stream"
---

[https://leetcode.com/problems/find-median-from-data-stream/](https://leetcode.com/problems/find-median-from-data-stream/) hard

To find the median we can maintain two heaps: smaller and larger. One decreasing and one increasing.
Peeking the top from those heaps will give us the median.

```

    //  [5 2 0] [6 7 10]
    //  dec     inc
    //   ^ peek  ^ peek

```

```kotlin

class MedianFinder() {
    val queDec = PriorityQueue<Int>(reverseOrder())
    val queInc = PriorityQueue<Int>()
    fun addNum(num: Int) {
        if (queDec.size == queInc.size) {
            queInc.add(num)
            queDec.add(queInc.poll())
        } else {
            queDec.add(num)
            queInc.add(queDec.poll())
        }
    }

    fun findMedian(): Double = if (queInc.size == queDec.size)
            (queInc.peek() + queDec.peek()) / 2.0
        else
            queDec.peek().toDouble()
}

```

Complexity: O(NlogN)
Memory: O(N)
