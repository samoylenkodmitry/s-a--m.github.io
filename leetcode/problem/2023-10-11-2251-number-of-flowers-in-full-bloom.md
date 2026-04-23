---
layout: leetcode-entry
title: "2251. Number of Flowers in Full Bloom"
permalink: "/leetcode/problem/2023-10-11-2251-number-of-flowers-in-full-bloom/"
leetcode_ui: true
entry_slug: "2023-10-11-2251-number-of-flowers-in-full-bloom"
---

[2251. Number of Flowers in Full Bloom](https://leetcode.com/problems/number-of-flowers-in-full-bloom/description/) hard
[blog post](https://leetcode.com/problems/number-of-flowers-in-full-bloom/solutions/4155880/kotlin-treemap-binary-search/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11102023-2251-number-of-flowers-in?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/cbc93f5c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/366

#### Problem TLDR

Array of counts of segments in intersection

#### Intuition

We need to quickly count how many segments are for any particular time. If we sort segments by `from` position, we can use line sweep, and we also need to track times when count decreases.
To find out how many `people` in a time range we can sort them and use Binary Search.

#### Approach

* to track count changes let's store time `delta`s in `timeToDelta` HashMap
* careful with storing decreases, they are starting in `to + 1`
* instead of sorting the segments we can use a `TreeMap`
* we need to preserve `people`s order, so use separate sorted `indices` collection

For better Binary Search code:
* use inclusive `lo` and `hi`
* check the last condition `lo == hi`
* always save a good result `peopleIndBefore = max(.., mid)`
* always move the borders `lo = mid + 1`, `hi = mid - 1`
* if `mid` is less than `target` drop everything on the left side: `lo = mid + 1`

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun fullBloomFlowers(flowers: Array<IntArray>, people: IntArray): IntArray {
      val peopleInds = people.indices.sortedBy { people[it] }
      val timeToDelta = TreeMap<Int, Int>()
      for ((from, to) in flowers) {
        timeToDelta[from] = 1 + (timeToDelta[from] ?: 0)
        timeToDelta[to + 1] = -1 + (timeToDelta[to + 1] ?: 0)
      }
      val res = IntArray(people.size)
      var count = 0
      var lastPeopleInd = -1
      timeToDelta.onEach { (time, delta) ->
        var lo = max(0, lastPeopleInd - 1)
        var hi = peopleInds.lastIndex
        var peopleIndBefore = -1
        while (lo <= hi) {
          val mid = lo + (hi - lo) / 2
          if (people[peopleInds[mid]] < time) {
            peopleIndBefore = max(peopleIndBefore, mid)
            lo = mid + 1
          } else hi = mid - 1
        }
        for (i in max(0, lastPeopleInd)..peopleIndBefore) res[peopleInds[i]] = count
        count += delta
        lastPeopleInd = peopleIndBefore + 1
      }
      return res
    }

```

