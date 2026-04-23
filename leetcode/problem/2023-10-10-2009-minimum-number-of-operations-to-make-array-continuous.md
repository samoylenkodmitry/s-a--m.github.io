---
layout: leetcode-entry
title: "2009. Minimum Number of Operations to Make Array Continuous"
permalink: "/leetcode/problem/2023-10-10-2009-minimum-number-of-operations-to-make-array-continuous/"
leetcode_ui: true
entry_slug: "2023-10-10-2009-minimum-number-of-operations-to-make-array-continuous"
---

[2009. Minimum Number of Operations to Make Array Continuous](https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous/description/) hard
[blog post](https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous/solutions/4152344/kotlin-binary-search/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10102023-2009-minimum-number-of-operations?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/92c8fe98.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/365

#### Problem TLDR

Min replacements to make array continuous `a[i] = a[i - 1] + 1`

#### Intuition

Use hint.
There are some ideas to solve this:
* if we choose any particular number from the array, we know how the result array must look like - `1 3 4 -> 1 2 3 or 3 4 5 or 4 5 6`
* we can sort the array and discard all numbers left to the current and right to the last of the result. For example, `1 3 4`, if current number is `1` we drop all numbers bigger than `3` as `1 2 3` is a result.
* to find the position of the right border, we can use a Binary Search
* now we have a range of numbers that almost good, but there can be `duplicates`. To count how many duplicates in range in O(1) we can precompute a prefix counter of the unique numbers.

#### Approach

Look at someone else's solution.
For better Binary Search code:
* use inclusive `lo` and `hi`
* check the last condition `lo == hi`
* always move the border `lo = mid + 1`, `hi = mid - 1`
* always update the result `toPos = min(toPos, mid)`
* choose which border to move by discarding not relevant `mid` position: `if nums[mid] is less than target, we can drop all numbers to the left, so move lo`

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minOperations(nums: IntArray): Int {
      nums.sort()
      val uniqPrefix = IntArray(nums.size) { 1 }
      for (i in 1..<nums.size) {
        uniqPrefix[i] = uniqPrefix[i - 1]
        if (nums[i] != nums[i - 1]) uniqPrefix[i]++
      }
      var minOps = nums.size - 1
      for (i in nums.indices) {
        val from = nums[i]
        val to = from + nums.size - 1
        var lo = i
        var hi = nums.size - 1
        var toPos = nums.size
        while (lo <= hi) {
          val mid = lo + (hi - lo) / 2
          if (nums[mid] > to) {
            toPos = min(toPos, mid)
            hi = mid - 1
          } else lo = mid + 1
        }
        val uniqCount = max(0, uniqPrefix[toPos - 1] - uniqPrefix[i]) + 1
        minOps = min(minOps, nums.size - uniqCount)
      }
      return minOps
    }

```

