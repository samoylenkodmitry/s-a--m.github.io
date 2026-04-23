---
layout: leetcode-entry
title: "1631. Path With Minimum Effort"
permalink: "/leetcode/problem/2023-09-16-1631-path-with-minimum-effort/"
leetcode_ui: true
entry_slug: "2023-09-16-1631-path-with-minimum-effort"
---

[1631. Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/description/) medium
[blog post](https://leetcode.com/problems/path-with-minimum-effort/solutions/4049798/kotln-a/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16092023-1631-path-with-minimum-effort?r=2bam17&utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/8e484fa1.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/341

#### Problem TLDR

Minimum absolute difference in path top-left to right-bottom

#### Intuition

To find an optimal path using some condition, we can use A* algorithm:
* add node to `PriorityQueue`
* choose the "optimal" one
* calculate a new heuristic for siblings and add to `PQ`

#### Approach

* use directions sequence for more clean code

#### Complexity

- Time complexity:
$$O(nmlog(nm))$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    val dirs = sequenceOf(1 to 0, 0 to 1, 0 to -1, -1 to 0)
    fun minimumEffortPath(heights: Array<IntArray>): Int {
      val pq = PriorityQueue<Pair<Pair<Int, Int>, Int>>(compareBy { it.second })
      pq.add(0 to 0 to 0)
      val visited = HashSet<Pair<Int, Int>>()
      while (pq.isNotEmpty()) {
        val (xy, diff) = pq.poll()
        if (!visited.add(xy)) continue
        val (x, y) = xy
        if (x == heights[0].lastIndex && y == heights.lastIndex) return diff
        dirs.map { (dx, dy) -> x + dx to y + dy }
          .filter { (x1, y1) -> x1 in 0..<heights[0].size && y1 in 0..<heights.size }
          .forEach { (x1, y1) -> pq.add(x1 to y1 to maxOf(diff, abs(heights[y][x] - heights[y1][x1]))) }
      }
      return 0
    }

```

