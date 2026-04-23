---
layout: leetcode-entry
title: "2369. Check if There is a Valid Partition For The Array"
permalink: "/leetcode/problem/2023-08-13-2369-check-if-there-is-a-valid-partition-for-the-array/"
leetcode_ui: true
entry_slug: "2023-08-13-2369-check-if-there-is-a-valid-partition-for-the-array"
---

[2369. Check if There is a Valid Partition For The Array](https://leetcode.com/problems/check-if-there-is-a-valid-partition-for-the-array/description/) medium
[blog post](https://leetcode.com/problems/check-if-there-is-a-valid-partition-for-the-array/solutions/3902038/kotlin-dfs-cache/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13082023-2369-check-if-there-is-a?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/7eabd6aa.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/307

#### Problem TLDR

Is it possible to partition an array of `2` or `3` equal nums or `3` increasing nums.

#### Intuition

Hint: don't spend much time trying to write a greedy solution.

We can consider every suffix of an array and make it a subproblem. Given it depends on only of the starting position, it can be safely cached.

#### Approach

* use Depth-First search and a HashMap for cache by position

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```

    fun validPartition(nums: IntArray): Boolean {
      val cache = mutableMapOf<Int, Boolean>()
      fun dfs(pos: Int): Boolean = cache.getOrPut(pos) {
        if (pos == nums.size) true
        else if (pos + 1 > nums.lastIndex) false
        else {
          val diff1 = nums[pos + 1] - nums[pos]
          if (diff1 == 0 && dfs(pos + 2)) true
          else if (pos + 2 > nums.lastIndex) false
          else {
            val diff2 = nums[pos + 2] - nums[pos + 1]
            (diff1 == 0 && diff2 == 0 || diff1 == 1 && diff2 == 1) && dfs(pos + 3)
          }
        }
      }
      return dfs(0)
    }

```

