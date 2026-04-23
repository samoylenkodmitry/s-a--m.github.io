---
layout: leetcode-entry
title: "2616. Minimize the Maximum Difference of Pairs"
permalink: "/leetcode/problem/2023-08-09-2616-minimize-the-maximum-difference-of-pairs/"
leetcode_ui: true
entry_slug: "2023-08-09-2616-minimize-the-maximum-difference-of-pairs"
---

[2616. Minimize the Maximum Difference of Pairs](https://leetcode.com/problems/minimize-the-maximum-difference-of-pairs/description/) medium
[blog post](https://leetcode.com/problems/minimize-the-maximum-difference-of-pairs/submissions/1016264480/)
[substack](https://dmitriisamoilenko.substack.com/p/09082023-2616-minimize-the-maximum?sd=pf)

![image.png](/assets/leetcode_daily_images/04dcd0eb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/303

#### Problem TLDR

Minimum of maximums possible `p` diffs of distinct array positions

#### Intuition

The `hint` is misleading, given the problem size 10^5 DP approach will give TLE, as it is n^2.

The real hint is:
* given the difference `diff`, how many pairs there are in an array, where `pair_diff <= diff`?
* if we increase the picked `diff` will that number grow or shrink?

Using this hint, we can solve the problem with Binary Search, as with growth of `diff`, there is a flip of when we can take `p` numbers and when we can't.

When counting the diffs, we use `Greedy` approach, and take the first possible, skipping its sibling. This will work, because we're answering the questions of `how many` rather than `maximum/minimum`.

#### Approach

For more robust Binary Search, use:
* inclusive `lo`, `hi`
* last condition `lo == hi`
* result: `if (count >= p) res = minOf(res, mid)`
* move border `lo = mid + 1`, `hi = mid - 1`

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minimizeMax(nums: IntArray, p: Int): Int {
        nums.sort()
        var lo = 0
        var hi = nums.last() - nums.first()
        var res = hi
        while (lo <= hi) {
          val mid = lo + (hi - lo) / 2
          var i = 1
          var count = 0
          while (i < nums.size) if (nums[i] - nums[i - 1] <= mid) {
            i += 2
            count++
          } else i++
          if (count >= p) res = minOf(res, mid)
          if (count >= p) hi = mid - 1 else lo = mid + 1
        }
        return res
    }

```

