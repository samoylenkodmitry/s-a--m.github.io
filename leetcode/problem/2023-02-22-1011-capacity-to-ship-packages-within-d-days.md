---
layout: leetcode-entry
title: "1011. Capacity To Ship Packages Within D Days"
permalink: "/leetcode/problem/2023-02-22-1011-capacity-to-ship-packages-within-d-days/"
leetcode_ui: true
entry_slug: "2023-02-22-1011-capacity-to-ship-packages-within-d-days"
---

[1011. Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/description/) medium

[blog post](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/solutions/3217409/kotlin-binary-search/)

```kotlin

fun shipWithinDays(weights: IntArray, days: Int): Int {
  var lo = weights.max()!!
  var hi = weights.sum()!!
  fun canShip(weight: Int): Boolean {
    var curr = 0
    var count = 1
    weights.forEach {
      curr += it
      if (curr > weight) {
        curr = it
        count++
      }
    }
    if (curr > weight) count++
    return count <= days
  }
  var min = hi
  while (lo <= hi) {
    val mid = lo + (hi - lo) / 2
    val canShip = canShip(mid)
    if (canShip) {
      min = minOf(min, mid)
      hi = mid - 1
    } else lo = mid + 1
  }
  return min
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/126
#### Intuition
Of all the possible capacities, there is an increasing possibility to carry the load. It may look like this: `not possible`, `not possible`, .., `not possible`, `possible`, `possible`, .., `possible`. We can binary search in that sorted space of possibilities.

#### Approach
To more robust binary search code:
* use inclusive `lo` and `hi`
* check the last case `lo == hi`
* check target condition separately `min = minOf(min, mid)`
* always move boundaries `lo` and `hi`
#### Complexity
- Time complexity:
  $$O(nlog_2(n))$$
- Space complexity:
  $$O(1)$$

