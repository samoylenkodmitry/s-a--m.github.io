---
layout: leetcode-entry
title: "1584. Min Cost to Connect All Points"
permalink: "/leetcode/problem/2023-09-15-1584-min-cost-to-connect-all-points/"
leetcode_ui: true
entry_slug: "2023-09-15-1584-min-cost-to-connect-all-points"
---

[1584. Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/description/) medium
[blog post](https://leetcode.com/problems/min-cost-to-connect-all-points/solutions/4046178/kotlin-priority-queue/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15092023-1584-min-cost-to-connect?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/65a0f07a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/340

#### Problem TLDR

Min manhatten distance connected graph

#### Intuition

We can start from any points, for example, `0`. Next, we must iterate over all possible edges and find one with minimum `distance`.

#### Approach

* use `Priority Queue` to sort all edges by distance
* we can stop after all nodes are visited once
* we can consider only the last edge distance in path

#### Complexity
- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun minCostConnectPoints(points: Array<IntArray>): Int {
      fun dist(from: Int, to: Int) =
        abs(points[from][0] - points[to][0]) + abs(points[from][1] - points[to][1])
      val notVisited = points.indices.toMutableSet()
      val pq = PriorityQueue<Pair<Int, Int>>(compareBy({ it.second }))
      pq.add(0 to 0)
      var sum = 0
      while (notVisited.isNotEmpty()) {
        val curr = pq.poll()
        if (!notVisited.remove(curr.first)) continue
        sum += curr.second
        for (to in notVisited) pq.add(to to dist(curr.first, to))
      }
      return sum
    }

```

