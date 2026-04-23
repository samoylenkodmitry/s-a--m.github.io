---
layout: leetcode-entry
title: "2366. Minimum Replacements to Sort the Array"
permalink: "/leetcode/problem/2023-08-30-2366-minimum-replacements-to-sort-the-array/"
leetcode_ui: true
entry_slug: "2023-08-30-2366-minimum-replacements-to-sort-the-array"
---

[2366. Minimum Replacements to Sort the Array](https://leetcode.com/problems/minimum-replacements-to-sort-the-array/description/) hard
[blog post](https://leetcode.com/problems/minimum-replacements-to-sort-the-array/solutions/3979280/kotlin-greedy/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30082023-2366-minimum-replacements?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/aac351ba.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/324

#### Problem TLDR

Minimum number of number splits to make an array non-decreasing

#### Intuition

The first idea is, if we walk the array backwards, suffix is a maximum number. The second idea is how to split the current number optimally. Consider example:

```
        // 3  8   3
        // 3  53  3 +1 split
        // 3  233 3 +1 split
        // 12 233 3 +1 split
```
We shall not split `8` into numbers bigger than `3`, so keep extracting them, until some remainder reached.
However, this will not be the case for another example: `2 9 4`, when we split `9` -> `5 + 4`, we should not split `5` into `1 + 4`, but `2 + 3`, but optimal split is `3 + 3 + 3`, as `3 < 4` and `3 > 2`.
Another strategy is to consider how many split operations we should do: `9 / 4 = 2`, then we know the number of parts: `9 = (x split y split z) = 3 + 3 + 3`. Each part is guaranteed to be less than `4` but the maximum possible to sum up to `9`.

#### Approach

* explicitly write the corner cases to simplify the thinking: ` x < prev, x == prev, prev == 1, x % prev == 0`
* give a meaningful variable names and don't prematurely simplify the math
* try to find the good example to debug the code

#### Complexity
- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```

    fun minimumReplacement(nums: IntArray): Long {
        if (nums.isEmpty()) return 0L
        // 3  8   3
        // 3  53  3 +1 split
        // 3  233 3 +1 split
        // 12 233 3 +1 split
        var prev = nums.last()
        var count = 0L
        for (i in nums.lastIndex downTo 0) {
            if (nums[i] == prev) continue
            if (nums[i] < prev) prev = nums[i]
            else if (prev == 1) count += nums[i] - 1
            else if (nums[i] % prev == 0) count += (nums[i] / prev) - 1
            else {
                val splits = nums[i] / prev // 15 / 4 = 3
                count += splits
                val countParts = splits + 1 // 4 = (3 4 4 4)
                prev = nums[i] / countParts // 15 / 4 = 3
            }
        }
        return count
    }

```

