---
layout: leetcode-entry
title: "1095. Find in Mountain Array"
permalink: "/leetcode/problem/2023-10-12-1095-find-in-mountain-array/"
leetcode_ui: true
entry_slug: "2023-10-12-1095-find-in-mountain-array"
---

[1095. Find in Mountain Array](https://leetcode.com/problems/find-in-mountain-array/description/) hard
[blog post](https://leetcode.com/problems/find-in-mountain-array/solutions/4159347/kotlin-3-binary-searches/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12102023-1095-find-in-mountain-array?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/1ef87b8b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/367

#### Problem TLDR

Binary Search in a mountain

#### Intuition

First, find the top of the slope. Next, do two Binary Searches on the left and on the right slopes

#### Approach

* to find a top search for where the increasing slope ends

For better Binary Search code
* use inclusive `lo` and `hi`
* check the last condition `lo == hi`
* always update the result `top = max(top, mid)`
* always move the borders `lo = mid + 1`, `hi = mid - 1`
* move border that cuts off the irrelevant part of the array

#### Complexity

- Time complexity:
$$O(log(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun findInMountainArray(target: Int, mountainArr: MountainArray): Int {
      var lo = 1
      var hi = mountainArr.length() - 1
      var top = -1
      while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        if (mountainArr.get(mid - 1) < mountainArr.get(mid)) {
          top = max(top, mid)
          lo = mid + 1
        } else hi = mid - 1
      }
      lo = 0
      hi = top
      while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        val m = mountainArr.get(mid)
        if (m == target) return mid
        if (m < target) lo = mid + 1 else hi = mid - 1
      }
      lo = top
      hi = mountainArr.length() - 1
      while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        val m = mountainArr.get(mid)
        if (m == target) return mid
        if (m < target) hi = mid - 1 else lo = mid + 1
      }
      return -1
    }

```

