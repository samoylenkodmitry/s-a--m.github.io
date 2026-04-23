---
layout: leetcode-entry
title: "1870. Minimum Speed to Arrive on Time"
permalink: "/leetcode/problem/2023-07-26-1870-minimum-speed-to-arrive-on-time/"
leetcode_ui: true
entry_slug: "2023-07-26-1870-minimum-speed-to-arrive-on-time"
---

[1870. Minimum Speed to Arrive on Time](https://leetcode.com/problems/minimum-speed-to-arrive-on-time/description/) medium
[blog post](https://leetcode.com/problems/minimum-speed-to-arrive-on-time/solutions/3817165/kotlin-binary-search/)
[substack](https://dmitriisamoilenko.substack.com/p/26072023-1870-minimum-speed-to-arrive?sd=pf)
![image.png](/assets/leetcode_daily_images/afb8f6b9.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/287

#### Problem TLDR

Max `speed` for all `dist` departing at round hours, be fit in `hour`

#### Intuition

Given the speed, we can calculate the `travel time` in O(n). With decreasing speed the time grows, so we can do the Binary Search

#### Approach

For more robust Binary Search code:
* use inclusive `lo` and `hi`
* check the last condition `lo == hi`
* always move the borders `lo = mid + 1`, `hi = mid - 1`
* always save the result `res = mid`

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```

    fun minSpeedOnTime(dist: IntArray, hour: Double): Int {
        var lo = 1
        var hi = 1_000_000_000
        var res = -1
        while (lo <= hi) {
          val mid = lo + (hi - lo) / 2
          var dt = 0.0
          val time = dist.fold(0.0) { r, t ->
            r + Math.ceil(dt).also { dt = t / mid.toDouble() }
          } + dt
          if (hour >= time) {
            res = mid
            hi = mid - 1
          } else lo = mid + 1
        }
        return res
    }

```

