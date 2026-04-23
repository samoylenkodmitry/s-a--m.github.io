---
layout: leetcode-entry
title: "2279. Maximum Bags With Full Capacity of Rocks"
permalink: "/leetcode/problem/2022-12-27-2279-maximum-bags-with-full-capacity-of-rocks/"
leetcode_ui: true
entry_slug: "2022-12-27-2279-maximum-bags-with-full-capacity-of-rocks"
---

[2279. Maximum Bags With Full Capacity of Rocks](https://leetcode.com/problems/maximum-bags-with-full-capacity-of-rocks/description/) medium

[https://t.me/leetcode_daily_unstoppable/65](https://t.me/leetcode_daily_unstoppable/65)

[blog post](https://leetcode.com/problems/maximum-bags-with-full-capacity-of-rocks/solutions/2957374/kotlin-sort-greedy/)

```kotlin
    fun maximumBags(capacity: IntArray, rocks: IntArray, additionalRocks: Int): Int {
       val inds = Array<Int>(capacity.size) { it }
       inds.sortWith(Comparator { a,b -> capacity[a]-rocks[a] - capacity[b] + rocks[b] })
       var rocksRemain = additionalRocks
       var countFull = 0
       for (i in 0..inds.lastIndex) {
           val toAdd = capacity[inds[i]] - rocks[inds[i]]
           if (toAdd > rocksRemain) break
           rocksRemain -= toAdd
           countFull++
       }
       return countFull
    }

```

We can logically deduce that the optimal solution is to take first bags with the smallest empty space.
Make an array of indexes and sort it by difference between `capacity` and `rocks`. Then just simulate rocks addition to each bug from the smallest empty space to the largest.

Space: O(n), Time: O(nlogn)

