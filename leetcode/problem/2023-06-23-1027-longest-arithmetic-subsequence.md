---
layout: leetcode-entry
title: "1027. Longest Arithmetic Subsequence"
permalink: "/leetcode/problem/2023-06-23-1027-longest-arithmetic-subsequence/"
leetcode_ui: true
entry_slug: "2023-06-23-1027-longest-arithmetic-subsequence"
---

[1027. Longest Arithmetic Subsequence](https://leetcode.com/problems/longest-arithmetic-subsequence/description/) medium
[blog post](https://leetcode.com/problems/longest-arithmetic-subsequence/solutions/3673731/kotlin-hard-problem-n-3/)
[substack](https://dmitriisamoilenko.substack.com/p/23062023-1027-longest-arithmetic?sd=pf)
![image.png](/assets/leetcode_daily_images/c171e832.webp)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/254

#### Problem TLDR
Max arithmetic subsequence length in array
#### Intuition
This was a hard problem for me :)
Naive Dynamic Programming solution with recursion and cache will give TLE.
Let's observe the result, adding numbers one-by-one:

```

// 20 1 15 3 10 5 8
// 20
// 20 1
//  1
// 20 20  1 15
//  1 15 15
//
// 20 20 20  1 1 15 3
// 1  15  3 15 3 3
//
// 20 20 20 20  1 1  1 15 15 10
//  1 15  3 10 15 3 10  3 10
//    10
//
// 20 20 20 20 20  1 1  1 1 15 15 15 10 5
//  1 15  3 10  5 15 3 10 5  3 10  5  5
//    10                        5
//     5
//
// 20 20 20 20 20 20  1 1  1 1 1 15 15 15 15 10 10 5 8
//  1 15  3 10  5  8 15 3 10 5 8  3 10  5  8  5  8 8
//    10                             5

```

For each pair `from-to` there is a sequence. When adding another number, we know what `next` numbers are expected.

#### Approach
We can put those sequences in a `HashMap` by `next` number key.
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

data class R(var next: Int, val d: Int, var size: Int)
fun longestArithSeqLength(nums: IntArray): Int {
    // 20 1 15 3 10 5 8
    // 20
    // 20 1
    //  1
    // 20 20  1 15
    //  1 15 15
    //
    // 20 20 20  1 1 15 3
    // 1  15  3 15 3 3
    //
    // 20 20 20 20  1 1  1 15 15 10
    //  1 15  3 10 15 3 10  3 10
    //    10
    //
    // 20 20 20 20 20  1 1  1 1 15 15 15 10 5
    //  1 15  3 10  5 15 3 10 5  3 10  5  5
    //    10                        5
    //     5
    //
    // 20 20 20 20 20 20  1 1  1 1 1 15 15 15 15 10 10 5 8
    //  1 15  3 10  5  8 15 3 10 5 8  3 10  5  8  5  8 8
    //    10                             5

    val nextToR = mutableMapOf<Int, MutableList<R>>()
        var max = 2
        nums.forEachIndexed { to, num ->
            nextToR.remove(num)?.forEach { r ->
                r.next = num + r.d
                max = maxOf(max, ++r.size)
                nextToR.getOrPut(r.next) { mutableListOf() } += r
            }
            for (from in 0..to - 1) {
                val d = num - nums[from]
                val next = num + d
                nextToR.getOrPut(next) { mutableListOf() } += R(next, d, 2)
            }
        }
        return max
    }

```

