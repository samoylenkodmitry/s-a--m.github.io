---
layout: leetcode-entry
title: "446. Arithmetic Slices II - Subsequence"
permalink: "/leetcode/problem/2022-11-27-446-arithmetic-slices-ii-subsequence/"
leetcode_ui: true
entry_slug: "2022-11-27-446-arithmetic-slices-ii-subsequence"
---

[446. Arithmetic Slices II - Subsequence](https://leetcode.com/problems/arithmetic-slices-ii-subsequence/) hard

[https://t.me/leetcode_daily_unstoppable/33](https://t.me/leetcode_daily_unstoppable/33)

```kotlin

    fun numberOfArithmeticSlices(nums: IntArray): Int {
        // 0 1 2 3 4 5
        // 1 2 3 1 2 3                diff = 1
        //   ^     ^ *                dp[5][diff] =
        //   |     |  \__ curr        1 + dp[1][diff] +
        //  prev   |                  1 + dp[4][diff]
        //        prev
        //
        val dp = Array(nums.size) { mutableMapOf<Long, Long> () }
        for (curr in 0..nums.lastIndex) {
            for (prev in 0 until curr) {
                val diff = nums[curr].toLong() - nums[prev].toLong()
                dp[curr][diff] = 1 + (dp[curr][diff]?:0L) + (dp[prev][diff]?:0L)
            }
        }
        return dp.map { it.values.sum()!! }.sum().toInt() - (nums.size)*(nums.size-1)/2
    }

```

dp[i][d] is the number of subsequences in range [0..i] with difference = d

```kotlin

array: "1 2 3 1 2 3"
For items  1  2  curr = 2:
diff = 1,  dp = 1
For items  1  2  3  curr = 3:
diff = 2,  dp = 1
diff = 1,  dp = 2
For items  1  2  3  1  curr = 1:
diff = 0,  dp = 1
diff = -1,  dp = 1
diff = -2,  dp = 1
For items  1  2  3  1  2  curr = 2:
diff = 1,  dp = 2
diff = 0,  dp = 1
diff = -1,  dp = 1
For items  1  2  3  1  2  3  curr = 3:
diff = 2,  dp = 2
diff = 1,  dp = 5
diff = 0,  dp = 1

```

and finally, we need to subtract all the sequences of length 2 and 1,
count of them is (n)*(n-1)/2

O(N^2) time, O(N^2) space

