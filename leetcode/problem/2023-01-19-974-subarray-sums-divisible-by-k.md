---
layout: leetcode-entry
title: "974. Subarray Sums Divisible by K"
permalink: "/leetcode/problem/2023-01-19-974-subarray-sums-divisible-by-k/"
leetcode_ui: true
entry_slug: "2023-01-19-974-subarray-sums-divisible-by-k"
---

[974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/) medium

[https://t.me/leetcode_daily_unstoppable/90](https://t.me/leetcode_daily_unstoppable/90)

[blog post](https://leetcode.com/problems/subarray-sums-divisible-by-k/solutions/3073473/kotlin-prefix-sum-and-remainders/)

```kotlin
    fun subarraysDivByK(nums: IntArray, k: Int): Int {
        // 4 5 0 -2 -3 1    k=5   count
        // 4                4:1   0
        //   9              4:2   +1
        //     9            4:3   +2
        //       7          2:1
        //          4       4:4   +3
        //             5    0:2   +1
        // 2 -2 2 -4       k=6
        // 2               2:1
        //    0            0:2    +1
        //      2          2:2    +1
        //        -2       2:3    +2
        // 1 2 13 -2 3  k=7
        // 1
        //   3
        //     16
        //        14
        //          17 (17-1*7= 10, 17-2*7=3, 17-3*7=-4, 17-4*7 = -11)
        val freq = mutableMapOf<Int, Int>()
        freq[0] = 1
        var sum = 0
        var res = 0
        nums.forEach {
            sum += it
            var ind = (sum % k)
            if (ind < 0) ind += k
            val currFreq = freq[ind] ?: 0
            res += currFreq
            freq[ind] = 1 + currFreq
        }
        return res
    }

```

We need to calculate a running sum.
For every current sum, we need to find any subsumes that are divisible by k, so `sum[i]: (sum[i] - sum[any prev]) % k == 0`.
Or, `sum[i] % k == sum[any prev] % k`.
Now, we need to store all `sum[i] % k` values, count them and add to result.

We can save frequency in a map, or in an array [0..k], because all the values are from that range.

Space: O(N), Time: O(N)

